import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from src.core.constants.definitions import DataAugmentation, Mode


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes, channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)
        self.channels = channels

    def forward(self, series):
        x = series[:, 0, :] if self.channels == 1 else series[:, :, 0, :]
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


class PretrainHead(nn.Module):
    def __init__(self, d_model, n_classes, channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)
        self.channels = channels

    def forward(self, series):
        x = self.layer_norm(series)
        x = self.fc(x)
        return x[:, 1:, :] if self.channels == 1 else x[:, :, 1:, :]

class ViT(nn.Module):
    # Defines the modules to be transformed to a low-rank adaption by LoRA
    lora_modules = ["to_patch_embedding.1", "to_qkv", "to_out.0", "net.1", "net.4"]
    # Defines the modules to be fully fine-tuned during LoRA training
    ft_modules = ["mlp_head.fc"]

    def __init__(self, *, seq_len, patch_size, stride, num_classes, dim, depth, heads, mlp_dim, mode: Mode, channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = (max(seq_len, patch_size) - patch_size) // stride + 1

        self.patch_dim = patch_size
        self.d_model = dim
        self.channels = channels
        self.num_patches = num_patches
        self.mode = mode

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.mask_token = nn.Parameter(torch.randn(dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        if mode == Mode.CLASSIFICATION:
            self.mlp_head = ClassificationHead(d_model=dim, n_classes=num_classes, channels=channels)
        else:
            self.mlp_head = PretrainHead(d_model=dim, n_classes=num_classes, channels=channels)

    def forward(self, series, mask, inverted_mask, indices, augmentation, **kwargs):
        if self.channels == 1 and augmentation is DataAugmentation.dropout_aug_transformer:
            x = self.dropout_augmentation(series, mask, inverted_mask, indices)
        elif self.channels == 1 and augmentation is DataAugmentation.test_time_aug_transformer:
            x = self.test_time_augmentation(series, mask, inverted_mask, indices)
        elif self.channels == 1:
            x = self.all_leads(series, mask, inverted_mask)
        else:
            x = self.per_lead(series, mask, inverted_mask)
        return self.mlp_head(x)

    def dropout_augmentation(self, x, mask, inverted_mask, indices):
        # Get the positional embeddings for these indices and the cls token
        pos_embeddings = torch.cat([
            self.pos_embedding[0, 0, :].reshape(1, self.d_model),
            self.pos_embedding.squeeze()[1:][indices].reshape((-1, self.d_model))
        ], dim=0)

        x = torch.reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2], self.patch_dim))
        x = x[:, indices, :]
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        if self.mode == Mode.PRETRAIN:
            mask, inverted_mask = mask.reshape(b, n, 1), inverted_mask.reshape(b, n, 1)
            x = x * mask + self.mask_token.expand(x.shape) * inverted_mask

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')

        x += pos_embeddings
        x = self.dropout(x)

        return self.transformer(x)

    def test_time_augmentation(self, x, mask, inverted_mask, indices):
        x = x[:, :, indices, :]

        # Get the positional embeddings for these indices and the cls token
        pos_embeddings = torch.cat([
            self.pos_embedding[0, 0, :].reshape(1, self.d_model),
            self.pos_embedding.squeeze()[1:].reshape((x.shape[1], -1, self.d_model))[:, indices, :].reshape((-1, self.d_model))
        ], dim=0)

        x = torch.reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2], self.patch_dim))
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        if self.mode == Mode.PRETRAIN:
            mask, inverted_mask = mask.reshape(b, n, 1), inverted_mask.reshape(b, n, 1)
            x = x * mask + self.mask_token.expand(x.shape) * inverted_mask

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')

        x += pos_embeddings
        x = self.dropout(x)

        return self.transformer(x)

    def all_leads(self, x, mask, inverted_mask):
        x = torch.reshape(x, shape=(x.shape[0], self.num_patches, self.patch_dim))
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        if self.mode == Mode.PRETRAIN:
            mask, inverted_mask = mask.reshape(b, n, 1), inverted_mask.reshape(b, n, 1)
            x = x * mask + self.mask_token.expand(x.shape) * inverted_mask

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        return self.transformer(x)

    def per_lead(self, x, mask, inverted_mask):
        x = self.to_patch_embedding(x)

        b, n_vars, n, d = x.shape
        if self.mode == Mode.PRETRAIN:
            mask, inverted_mask = mask.reshape(b, n_vars, n, 1), inverted_mask.reshape(b, n_vars, n, 1)
            x = x * mask + self.mask_token.expand(x.shape) * inverted_mask

        cls_tokens = repeat(self.cls_token, 'd -> b n_var d', b=b, n_var=n_vars)
        x, ps = pack([cls_tokens, x], 'b n_var * d')

        x = torch.reshape(x, (b * n_vars, n+1, d))
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = torch.reshape(x, (b, n_vars, n+1, d))

        return x


# if __name__ == '__main__':
#
#     v = ViT(
#         seq_len = 256,
#         patch_size = 16,
#         num_classes = 1000,
#         dim = 1024,
#         depth = 6,
#         heads = 8,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )
#
#     time_series = torch.randn(4, 3, 256)
#     logits = v(time_series) # (4, 1000)