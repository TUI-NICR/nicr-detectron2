from torch.nn.functional import mse_loss as mse

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from ...utils.pth_checkpointer import PTHDetectionCheckpointer
from ...utils.grasping.loss import gnlll_loss, weighted_gnlll_loss
from .grasp_net import GraspNet
from nicr_detectron2.utils.grasping.loss.loss_utils import get_weights

import copy

import torch

@META_ARCH_REGISTRY.register()
class UncertainGraspNet(GraspNet):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mean_model = META_ARCH_REGISTRY.get(
                                cfg.MODEL.GRASP_NET.UNCERTAIN_NET)(cfg)

        # does the mean_model provide a var head or not?
        self.with_var_head = cfg.MODEL.GRASP_NET.VARIANCE_HEAD
        if self.with_var_head: # only implemented in GRConvNet
            assert cfg.MODEL.GRASP_NET.UNCERTAIN_NET in ["GRConvNet", "GGCNN2"], print("VARIANCE_HEAD not implemented!")
        # if no copy the mean model and use it as var model
        if not self.with_var_head:
            self.var_model = copy.deepcopy(self.mean_model)

        self.use_smoothed_loss = cfg.MODEL.GRASP_NET.USE_SMOOTHED_LOSS
        self.use_max_loss = cfg.MODEL.GRASP_NET.MAX_LOSS

    def load_pretrained_weights(self):
        # no weights for UncertainGraspNet given?
        # then check and load weights for the underlying architecture
        if self.cfg.MODEL.WEIGHTS == "" and self.cfg.MODEL.GRASP_NET.UNCERTAIN_NET_WEIGHTS != "":
            checkpointer = PTHDetectionCheckpointer(self, prefix="mean_model.") # nested loading with PTHDetectionCheckpointer
            checkpointer.load(self.cfg.MODEL.GRASP_NET.UNCERTAIN_NET_WEIGHTS)
            if not self.with_var_head:
                self.var_model = copy.deepcopy(self.mean_model)
                for param in self.mean_model.parameters():
                    param.requires_grad = False
                self.mean_model.eval()

            # disable MC Dropout for variance model
            self.var_model.mc_dropout = False

    def apply_network(self, modalities):
        self.var_model.mc_dropout = False
        # apply mean and variance model and stack outputs
        if self.with_var_head:
            out_dict = self.mean_model.apply_network(modalities)
        else:
            # apply var model first as mc dropout would change modalities
            var = self.var_model.apply_network(modalities)
            out_dict = self.mean_model.apply_network(modalities)
            var = {k.replace("out","var"): var[k] for k in var.keys()}
            out_dict.update(var)

        # for stability reasons: clip log(var) from -20 to 20
        for key in out_dict.keys():
            if "var" in key: out_dict[key] = torch.clip(out_dict[key], -20.0, 20.0)

        # if self.train:
        #     storage = get_event_storage()
        #     for key in out_dict:
        #         if 'var' in  key:
        #             storage.put_scalar(f'mean_{key}', torch.exp(out_dict[key].mean()))

        return out_dict

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

        for sample_idx in range(out_dict["pos_out"].shape[0]):
            # q_var_img = torch.exp(pos_out[1, sample_idx])
            # cos_var_img = torch.exp(cos_out[1, sample_idx])
            # sin_var_img = torch.exp(sin_out[1, sample_idx])
            # width_var_img = torch.exp(width_out[1, sample_idx])

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

    def compute_losses(self, batched_inputs, out_dict, **kwargs):

        pos_teacher = torch.stack([batched_input["pos_gt"].to(self.device) for batched_input in batched_inputs])
        cos_teacher = torch.stack([batched_input["cos_gt"].to(self.device) for batched_input in batched_inputs])
        sin_teacher = torch.stack([batched_input["sin_gt"].to(self.device) for batched_input in batched_inputs])
        width_teacher = torch.stack([batched_input["width_gt"].to(self.device) for batched_input in batched_inputs])


        if self.mask_loss or -1 in pos_teacher:
            # if we have explicit void labels
            if -1 in pos_teacher:
                # mask all pixels with -1 or 0 quality for angle and width
                valid_mask = pos_teacher > 0
                # 0 quality will be void
                valid_mask_pos = pos_teacher != 0
                # set -1 quality to 0 as these are the negative labels
                pos_teacher[pos_teacher == -1] = 0
            else:
                # if we only have labels between 0 and 1
                # only apply mask to angle and width where quality is 0
                # for quality we need all labels
                valid_mask = pos_teacher != 0
                valid_mask_pos = torch.ones_like(pos_teacher)
            do_masking = True
        else:
            do_masking = False
            valid_mask_pos = torch.ones_like(pos_teacher)
            valid_mask = torch.ones_like(pos_teacher)

        if self.train:
            storage = get_event_storage()
            for key in out_dict:
                if 'out' in key:
                    if 'pos' in key:
                        storage.put_scalar(key.replace('out', 'mse'), mse(out_dict[key] * valid_mask_pos, pos_teacher * valid_mask_pos))
                    elif 'cos' in key:
                        storage.put_scalar(key.replace('out', 'mse'), mse(out_dict[key] * valid_mask, cos_teacher * valid_mask))
                    elif 'sin' in key:
                        storage.put_scalar(key.replace('out', 'mse'), mse(out_dict[key] * valid_mask, sin_teacher * valid_mask))
                    elif 'width' in key:
                        storage.put_scalar(key.replace('out', 'mse'), mse(out_dict[key] * valid_mask, width_teacher * valid_mask))
                if 'var' in  key:
                    if 'pos' in key:
                        storage.put_scalar(f'mean_{key}', torch.exp(out_dict[key][valid_mask_pos == 1].mean()))
                    else:
                        storage.put_scalar(f'mean_{key}', torch.exp(out_dict[key][valid_mask == 1].mean()))
        if self.use_max_loss:
            assert "weights" in kwargs.keys()
            p_gaussian_loss = weighted_gnlll_loss(out_dict["pos_out"], pos_teacher, out_dict["pos_var"],
                                                  kwargs["weights"]["pos_weights"], use_smoothed=self.use_smoothed_loss)

            cos_gaussian_loss = weighted_gnlll_loss(out_dict["cos_out"], cos_teacher, out_dict["cos_var"],
                                                    kwargs["weights"]["cos_weights"], use_smoothed=self.use_smoothed_loss)

            sin_gaussian_loss = weighted_gnlll_loss(out_dict["sin_out"], sin_teacher, out_dict["sin_var"],
                                                    kwargs["weights"]["sin_weights"], use_smoothed=self.use_smoothed_loss)

            width_gaussian_loss = weighted_gnlll_loss(out_dict["width_out"], width_teacher, out_dict["width_var"],
                                                      kwargs["weights"]["width_weights"], use_smoothed=self.use_smoothed_loss)
        else:
            p_gaussian_loss = gnlll_loss(out_dict["pos_out"], pos_teacher, out_dict["pos_var"], use_smoothed=self.use_smoothed_loss,
                                         mask=valid_mask_pos if do_masking else None)
            cos_gaussian_loss = gnlll_loss(out_dict["cos_out"], cos_teacher, out_dict["cos_var"], use_smoothed=self.use_smoothed_loss,
                                         mask=valid_mask_pos if do_masking else None)
            sin_gaussian_loss = gnlll_loss(out_dict["sin_out"], sin_teacher, out_dict["sin_var"], use_smoothed=self.use_smoothed_loss,
                                         mask=valid_mask_pos if do_masking else None)
            width_gaussian_loss = gnlll_loss(out_dict["width_out"], width_teacher, out_dict["width_var"], use_smoothed=self.use_smoothed_loss,
                                         mask=valid_mask_pos if do_masking else None)

        # default trainer computes the sum of this dict
        losses = {
            "pos_loss": p_gaussian_loss,
            "cos_loss": cos_gaussian_loss,
            "sin_loss": sin_gaussian_loss,
            "width_loss": width_gaussian_loss
        }
        return losses

    def compute_weights(self, batched_inputs):
        modalities = self.preprocess_image(batched_inputs)
        out_dict = self.apply_network(modalities)

        pos_teacher = torch.stack([batched_input["pos_gt"].to(self.device) for batched_input in batched_inputs])
        cos_teacher = torch.stack([batched_input["cos_gt"].to(self.device) for batched_input in batched_inputs])
        sin_teacher = torch.stack([batched_input["sin_gt"].to(self.device) for batched_input in batched_inputs])
        width_teacher = torch.stack([batched_input["width_gt"].to(self.device) for batched_input in batched_inputs])

        if self.use_max_loss:
            pos_weights = get_weights(out_dict["pos_out"], torch.exp(out_dict["pos_var"]), pos_teacher, type="max")
            cos_weights = get_weights(out_dict["cos_out"], torch.exp(out_dict["cos_var"]), cos_teacher, type="")
            sin_weights = get_weights(out_dict["sin_out"], torch.exp(out_dict["sin_var"]), sin_teacher, type="")
            width_weights = get_weights(out_dict["width_out"], torch.exp(out_dict["width_var"]), width_teacher, type="")
        else:
            raise NotImplementedError
        # computing other loss weights goes here

        return {
            "pos_weights": pos_weights,
            "cos_weights": cos_weights,
            "sin_weights": sin_weights,
            "width_weights": width_weights,
        }
