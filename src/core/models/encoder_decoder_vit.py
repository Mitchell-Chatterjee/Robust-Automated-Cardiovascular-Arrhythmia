import copy

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from src.core.constants.definitions import Mode

"""
This model is taken from:

Hu, Rui, Jie Chen, and Li Zhou. "Spatiotemporal self-supervised representation learning from multi-lead ECG signals." 
Biomedical Signal Processing and Control 84 (2023): 104772.

If using this model, please cite the authors in your work. 

Unfortunately, we do not have access to the original code. Therefore, the model has been re-created using the 
model architecture specified in the above paper.

The dimensions of the model are hardcoded so they match those of the paper. Also the convolutional embedding steps have
been expanded to ensure clarity. Future work may wish to integrate the dimensionso of the model with our own work so 
they can be set dynamically. Moreover, it may be benificial to include the convolutional steps in a sequential wrapper
for brevity.
"""

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, series):
        x = series[:, 0, :]
        # x = self.layer_norm(x)
        x = self.fc(x)
        return x


class PretrainHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.transformer = Transformer(64, 3, 8, 64 // 8, 384, 0.1)
        self.linear2 = nn.Linear(64, 50)

    def forward(self, x):
        x = self.linear1(x)
        x = self.transformer(x)
        x = self.linear2(x)
        return x[:, 1:, :]


class EncoderDecoderViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, mode: Mode, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.mode = mode

        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=50, stride=50, dilation=2, padding=25)
        )

        self.ln = nn.LayerNorm(128)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.mask_token = nn.Parameter(torch.randn(dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        if mode == Mode.CLASSIFICATION:
            self.mlp_head = ClassificationHead(d_model=dim, n_classes=num_classes)
        else:
            self.mlp_head = PretrainHead(d_model=dim, n_classes=num_classes)

    def forward(self, series, mask, inverted_mask, **kwargs):

        x = torch.reshape(series, shape=(series.shape[0], 1, series.shape[1] * series.shape[2] * series.shape[3]))

        x = self.to_patch_embedding(x)
        x = x.transpose(2, 1)
        x = self.ln(x)

        b, n, d = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # Encoder only gets the unmasked tokens
        if self.mode == Mode.PRETRAIN:
            mask, inverted_mask = mask.reshape(b, -1), inverted_mask.reshape(b, -1)
            mask = torch.cat([repeat(torch.tensor([True]).to('cuda:0'), 'd -> b d', b=b), mask], dim=1)
            inverted_mask = torch.cat([repeat(torch.tensor([False]).to('cuda:0'), 'd -> b d', b=b), inverted_mask], dim=1)
            mask, inverted_mask = mask.reshape(b, n+1, 1), inverted_mask.reshape(b, n+1, 1)

            x_copy = x.clone()
            x = x[mask.reshape(b, -1)].reshape(b, -1, d)
        else:
            x += self.pos_embedding[:, :n+1]

        x = self.dropout(x)
        x = self.transformer(x)

        # Refill encoded values and masked tokens
        if self.mode == Mode.PRETRAIN:
            x_copy[mask.reshape(b, -1)] = x.reshape(shape=x_copy[mask.reshape(b, -1)].shape)
            x = x_copy

            x = x * mask + self.mask_token.expand(x.shape) * inverted_mask

            x += self.pos_embedding[:, :n+1]

        return self.mlp_head(x)

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
