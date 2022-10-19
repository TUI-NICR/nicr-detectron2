from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

import torch
from torch import nn

from .grasp_net import GraspNet
from ...utils.grasping.grconvnet_helpers import ResidualBlock

from collections import OrderedDict


@META_ARCH_REGISTRY.register()
class GRConvNet(GraspNet):

    def __init__(self, cfg):
        super().__init__(cfg)

        # TODO: config based
        if 'color' in self.modalities and 'depth' in self.modalities:
            input_channels = 4
        elif 'color' in self.modalities:
            input_channels = 3
        elif 'depth' in self.modalities:
            input_channels = 1
        else:
            raise NotImplementedError(
                'Unknown modalities: ' + ' '.join(self.modalities)
            )

        dropout_rate = cfg.MODEL.GRASP_NET.GRCONVNET.DROPOUT_RATE
        num_residual_blocks = cfg.MODEL.GRASP_NET.GRCONVNET.NUM_RES_BLOCKS

        if cfg.MODEL.GRASP_NET.GRCONVNET.TYPE_RES_BLOCK == "ResidualBlock":
            block = ResidualBlock
        else:
            raise NotImplementedError

        module_dict = OrderedDict()

        # input layer and first convolutions
        module_dict.update(
            {
                "conv0": nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4),
                "bn0": nn.BatchNorm2d(32),
                "act0": nn.ReLU(inplace=True),

                "dropout1": nn.Dropout2d(p=dropout_rate),
                "conv1":nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                "bn1": nn.BatchNorm2d(64),
                "act1": nn.ReLU(inplace=True),

                "dropout2": nn.Dropout2d(p=dropout_rate),
                "conv2":nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                "bn2": nn.BatchNorm2d(128),
                "act2": nn.ReLU(inplace=True)
            }
        )

        # residual blocks
        for i in range(num_residual_blocks):
            module_dict.update(
                {
                    f"res_block{i}": block(128, 128, dropout_rate=dropout_rate)
                }
            )

        # upsample
        module_dict.update(
            {
                "dropout_3": nn.Dropout2d(p=dropout_rate),
                "conv_tr0": nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
                "bn3": nn.BatchNorm2d(64),
                "act3": nn.ReLU(inplace=True),

                "dropout_4": nn.Dropout2d(p=dropout_rate),
                "conv_tr1": nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1),
                "bn4": nn.BatchNorm2d(32),
                "act4": nn.ReLU(inplace=True),

                "conv_tr2": nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4),
            }
        )
        self.features = nn.Sequential(module_dict)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        # TODO: check best initialization
        if self.with_var_head:
            self.pos_var = nn.Conv2d(32, 1, kernel_size=2)
            self.cos_var = nn.Conv2d(32, 1, kernel_size=2)
            self.sin_var = nn.Conv2d(32, 1, kernel_size=2)
            self.width_var = nn.Conv2d(32, 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def inference(self, out_dict):
        results = super().inference(out_dict)

        if self._do_gaussian_filter:
            out_dict["pos_out"] = self._gaussian_filter(out_dict["pos_out"])
            out_dict["pos_var"] = self._gaussian_filter(out_dict["pos_var"])
            out_dict["ang_out"] = self._gaussian_filter(torch.atan2(out_dict["sin_out"], out_dict["cos_out"]) / 2.0)
            out_dict["width_out"] = self._gaussian_filter(out_dict["width_out"]*150.)
            out_dict["width_var"] = self._gaussian_filter(out_dict["width_var"])
        else:
            out_dict["ang_out"] = torch.atan2(out_dict["sin_out"], out_dict["cos_out"]) / 2.0
            out_dict["width_out"] *= 150.

        for sample_idx in range(len(results)):

            # q_var_img = torch.exp(pos_out[1, sample_idx])
            # cos_var_img = torch.exp(cos_out[1, sample_idx])
            # sin_var_img = torch.exp(sin_out[1, sample_idx])
            # width_var_img = torch.exp(width_out[1, sample_idx])
            if self.with_var_head:
                results[sample_idx]['aleatoric'] = {
                    "cos":  torch.exp(out_dict["cos_var"][sample_idx]),
                    "pos":  torch.exp(out_dict["pos_var"][sample_idx]),
                    "sin":  torch.exp(out_dict["sin_var"][sample_idx]),
                    "width":  torch.exp(out_dict["width_var"][sample_idx])
                }


            if self.mc_dropout:
                results[sample_idx]['epistemic'] = {
                    "pos": out_dict["pos_out_epistemic"][sample_idx],
                    "cos": out_dict["cos_out_epistemic"][sample_idx],
                    "sin": out_dict["sin_out_epistemic"][sample_idx],
                    "width": out_dict["width_out_epistemic"][sample_idx]
                }

            # results.append(result)
        return results

    def apply_network(self, modalities):

        # inference
        if 'color' in self.modalities and 'depth' in self.modalities:
            input = torch.cat([modalities[0].tensor, modalities[1].tensor],dim=1)
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
            out_dict.update({
                "pos_var": pos_var,
                "cos_var": cos_var,
                "sin_var": sin_var,
                "width_var": width_var
            })

        if self.mc_dropout:
            if self.mc_dropout and self.training: print("WARNING: Using MC-Dropout during training is slow!")
            # setting this to False temporarily as compute_epistemic_uncertainty is calling apply_network
            # This would create an infinite loop
            self.mc_dropout = False
            epistemic_uncertainties = self.compute_epistemic_uncertainty(modalities=modalities, num_mc_samples=self.num_mc_samples)
            self.mc_dropout = True
            out_dict.update({
                k+"_epistemic": epistemic_uncertainties[k] for k in epistemic_uncertainties.keys()
            })

        return out_dict