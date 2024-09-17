from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.models.layers.revin import RevIN
from src.core.support.timeseries_transformations import *


def transformations_from_strings(transformations, t_params, n_leads, seq_len):
    if transformations is None or "None" in transformations:
        return None

    def str_to_trafo(trafo):
        if trafo == "ChannelResize":
            return TChannelResize(magnitude_range=t_params["t_magnitude_range"])
        elif trafo == "GaussianNoise":
            return TGaussianNoise(scale=t_params["t_gaussian_scale"])
        elif trafo == "SampledNoise":
            return TSampledNoise(t_params['t_root_path_noise'], n_leads)
        elif trafo == "NoTransform":
            return TNoTransform()
        elif trafo == "RandomResizedCrop":
            return TRandomResizedCrop(crop_ratio_range=t_params["t_rr_crop_ratio_range"], output_size=seq_len)
        # Unimplemented transformations are commented out
        # elif trafo == "Negation":
        #     return TNegation()
        # elif trafo == "DynamicTimeWarp":
        #     return TDynamicTimeWarp(warps=t_params["t_warps"], radius=t_params["t_radius"])
        # elif trafo == "DownSample":
        #     return TDownSample(downsample_ratio=t_params["t_downsample_ratio"])
        # elif trafo == "TimeWarp":
        #     return TimeWarp(epsilon=t_params["t_epsilon"])
        elif trafo == "TimeOut":
            return TTimeOut(crop_ratio_range=t_params["t_to_crop_ratio_range"])
        # elif trafo == "BaselineWander":
        #     return TBaselineWander(Cmax=t_params["t_bw_cmax"])
        # elif trafo == "PowerlineNoise":
        #     return TPowerlineNoise(Cmax=t_params["t_pl_cmax"])
        # elif trafo == "EMNoise":
        #     return TEMNoise(Cmax=t_params["t_em_cmax"])
        # elif trafo == "BaselineShift":
        #     return TBaselineShift(Cmax=t_params["t_bs_cmax"])
        # elif trafo == "GaussianBlur":
        #     return TGaussianBlur1d()
        # elif trafo == "Normalize":
        #     return TNormalize()
        else:
            raise Exception(str(trafo) + " is not a valid transformation")

    # for numpy transformations
    trafo_list = [str_to_trafo(trafo) for trafo in transformations]
    return trafo_list


class RevInSupport(AbstractSupportClass):
    def __init__(self, num_features: int, eps=1e-5,
                 affine: bool = False, denorm: bool = True, norm_stats: str = 'BN'):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This support only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine, norm_stats=norm_stats)

    def before_forward(self, xb):
        # Rev-in norm
        return self.revin(xb, 'norm')  # xb_revin: [bs x nvars x seq_len]

    def after_forward(self, pred):
        if self.denorm:
            # Rev-in denorm
            return self.revin(pred, 'denorm')  # pred: [bs x nvars x target_window]
        return pred
