from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

import torch
from torch import nn

from .grasp_net import GraspNet

@META_ARCH_REGISTRY.register()
class GGCNN2(GraspNet):

    def __init__(self, cfg):
        super().__init__(cfg)

        # TODO: config based
        # inference
        if 'color' in self.modalities and 'depth' in self.modalities:
            input_channels = 4
        elif 'color' in self.modalities:
            input_channels = 3
        elif 'depth' in self.modalities:
            input_channels = 1
        else:
            raise NotImplementedError
        filter_sizes=None
        l3_k_size=5
        dilations=None

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        dropout_rate = cfg.MODEL.GRASP_NET.GGCNN2.DROPOUT_RATE

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(inplace=True),

            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        self.with_var_head = cfg.MODEL.GRASP_NET.VARIANCE_HEAD
        # TODO: check best initialization
        if self.with_var_head:
            self.pos_var = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
            self.cos_var = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
            self.sin_var = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
            self.width_var = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    # @property
    # def device(self):
    #     return self.pixel_mean_color.device

    def apply_network(self, modalities):

        # inference
        if 'color' in self.modalities and 'depth' in self.modalities:
            input = torch.cat([modalities[0].tensor, modalities[1].tensor],dim=1)
        elif 'color' in self.modalities:
            input = modalities[0].tensor
        elif 'depth' in self.modalities:
            input = modalities[0].tensor
        else:
            input = modalities[0].tensor

        x = self.features(input)

        pos_out = self.pos_output(x)
        cos_out = self.cos_output(x)
        sin_out = self.sin_output(x)
        width_out = self.width_output(x)

        out_dict = {
            "pos_out": pos_out,
            "cos_out": cos_out,
            "sin_out": sin_out,
            "width_out": width_out
        }

        if self.with_var_head:
            pos_var = self.pos_var(x)
            cos_var = self.cos_var(x)
            sin_var = self.sin_var(x)
            width_var = self.width_var(x)
            out_dict['aleatoric'] = {
                "pos_var": pos_var,
                "cos_var": cos_var,
                "sin_var": sin_var,
                "width_var": width_var
            }

        return out_dict