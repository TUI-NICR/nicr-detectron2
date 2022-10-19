import numpy as np

from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

import torch
from torch import nn

from ...utils.grasping.loss import gnlll_loss


class ImageListWrapper:
    """Wrapper class for ImageList.
    This is used so that the base code of GernaralizedRCNN
    can be used with new format for images (dicts instead of images).
    """
    def __init__(self, image_list: ImageList) -> None:
        self._image_list = image_list

    def __getattr__(self, name: str):
        return getattr(self._image_list[0], name)

    def __iter__(self):
        return iter(self._image_list)

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, item):
        return self._image_list[item]

@META_ARCH_REGISTRY.register()
class GraspNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.modalities = cfg.INPUT.MODALITIES
        for i, mod in enumerate(self.modalities):
            self.register_buffer("pixel_mean_" + mod,
                                 torch.Tensor(cfg.MODEL.PIXEL_MEANS[i]).view(-1, 1, 1))
            self.register_buffer("pixel_std_" + mod,
                                 torch.Tensor(cfg.MODEL.PIXEL_STDS[i]).view(-1, 1, 1))

        if cfg.MODEL.GRASP_NET.USE_SMOOTHED_LOSS:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()

        self.mask_loss = cfg.MODEL.GRASP_NET.MASK_LOSS
        self.invert_mask_loss = cfg.MODEL.GRASP_NET.INVERT_MASK_LOSS

        self._gaussian_filter = self.get_gaussian_kernel()
        self._do_gaussian_filter = cfg.MODEL.GRASP_NET.GAUSS_POSTPROCESS

        self.mc_dropout = cfg.MODEL.GRASP_NET.MC_DROPOUT
        self.num_mc_samples = cfg.MODEL.GRASP_NET.MC_SAMPLES

        self.with_var_head = cfg.MODEL.GRASP_NET.VARIANCE_HEAD

    @property
    def device(self):
        if "color" in self.modalities:
            return self.pixel_mean_color.device
        elif "depth" in self.modalities:
            return self.pixel_mean_depth.device
        else:
            raise NotImplementedError

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        def _process_data(mod_i):
            name = self.modalities[mod_i]
            mean = getattr(self, "pixel_mean_" + str(name))
            std = getattr(self, "pixel_std_" + str(name))
            images = [x["image"][name].to(self.device).float() for x in batched_inputs]
            if name.lower().startswith('color'):
                images = [((x / 255) - mean) / std for x in images]
            elif name.lower().startswith('depth'):
                images2 = []
                for image in images:
                    image_0 = image == 0
                    image = (image - mean) / std
                    # set invalid values back to zero again
                    image[image_0] = 0
                    images2.append(image)
                images = images2
            images = ImageList.from_tensors(images, 0)
            return images

        return ImageListWrapper([_process_data(i) for i in range(len(self.modalities))])

    def apply_network(self, modalities):
        raise NotImplementedError

    def forward(self, batched_inputs, **kwargs):
        modalities = self.preprocess_image(batched_inputs)
        out_dict = self.apply_network(modalities)
        if not self.training:
            return self.inference(out_dict)

        return self.compute_losses(batched_inputs, out_dict, **kwargs)

    def compute_losses(self, batched_inputs, out_dict, **kwargs):
        pos_teacher = torch.stack([batched_input["pos_gt"].to(self.device)
                                   for batched_input in batched_inputs])

        cos_teacher = torch.stack([batched_input["cos_gt"].to(self.device)
                                   for batched_input in batched_inputs])

        sin_teacher = torch.stack([batched_input["sin_gt"].to(self.device)
                                   for batched_input in batched_inputs])

        width_teacher = torch.stack([batched_input["width_gt"].to(self.device)
                                     for batched_input in batched_inputs])

        # mask if we have explicit void labels or if config says so
        if self.mask_loss or -1 in pos_teacher:
            # if we have explicit void labels
            if -1 in pos_teacher:
                # mask all pixels with -1 or 0 quality for angle and width
                valid_mask = pos_teacher > 0
                if self.invert_mask_loss:
                    # -1 quality will be void
                    valid_mask_pos = pos_teacher != -1
                else:
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

            if self.with_var_head:
                pos_loss = gnlll_loss(
                    out_dict['pos_out'], pos_teacher,
                    out_dict['pos_var'], reduction=False)
                sin_loss = gnlll_loss(
                    out_dict['sin_out'], sin_teacher,
                    out_dict['sin_var'], reduction=False)
                cos_loss = gnlll_loss(
                    out_dict['cos_out'], cos_teacher,
                    out_dict['cos_var'], reduction=False)
                width_loss = gnlll_loss(
                    out_dict['width_out'], width_teacher,
                    out_dict['width_var'], reduction=False)

                pos_loss *= valid_mask_pos
                sin_loss *= valid_mask
                cos_loss *= valid_mask
                width_loss *= valid_mask

                pos_loss = torch.mean(pos_loss)
                sin_loss = torch.mean(sin_loss)
                cos_loss = torch.mean(cos_loss)
                width_loss = torch.mean(width_loss)
            else:
                pos_loss = self.criterion(
                    out_dict["pos_out"] * valid_mask_pos, pos_teacher * valid_mask_pos)
                cos_loss = self.criterion(
                    out_dict["cos_out"] * valid_mask, cos_teacher * valid_mask)
                sin_loss = self.criterion(
                    out_dict["sin_out"] * valid_mask, sin_teacher * valid_mask)
                width_loss = self.criterion(
                    out_dict["width_out"] * valid_mask, width_teacher * valid_mask)
        else:
            if self.with_var_head:
                pos_loss = gnlll_loss(
                    out_dict['pos_out'], pos_teacher, out_dict['pos_var'], reduction=True)
                sin_loss = gnlll_loss(
                    out_dict['sin_out'], sin_teacher, out_dict['sin_var'], reduction=True)
                cos_loss = gnlll_loss(
                    out_dict['cos_out'], cos_teacher, out_dict['cos_var'], reduction=True)
                width_loss = gnlll_loss(
                    out_dict['width_out'], width_teacher, out_dict['width_var'],
                    reduction=True)
            else:
                pos_loss = self.criterion(out_dict["pos_out"], pos_teacher)
                cos_loss = self.criterion(out_dict["cos_out"], cos_teacher )
                sin_loss = self.criterion(out_dict["sin_out"], sin_teacher )
                width_loss = self.criterion(out_dict["width_out"], width_teacher)

        # default trainer computes the sum of this dict
        losses = {
            "pos_loss": pos_loss,
            "cos_loss": cos_loss,
            "sin_loss": sin_loss,
            "width_loss": width_loss
        }
        return losses

    # From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3 pylint: disable=line-too-long
    @staticmethod
    def get_gaussian_kernel(kernel_size=-1, sigma=2, channels=1):

        if kernel_size == -1:
            kernel_size = int(4. * sigma + 0.5)

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*np.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False,
                                    padding='same')

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def inference(self, out_dict):
        results = []

        if self._do_gaussian_filter:
            out_dict["pos_out"] = self._gaussian_filter(out_dict["pos_out"])
            out_dict["ang_out"] = self._gaussian_filter(torch.atan2(out_dict["sin_out"], out_dict["cos_out"]) / 2.0)
            out_dict["width_out"] = self._gaussian_filter(out_dict["width_out"] * 150.)
        else:
            out_dict["ang_out"] = torch.atan2(out_dict["sin_out"], out_dict["cos_out"]) / 2.0
            out_dict["width_out"] *= 150.

        for sample_idx in range(out_dict["pos_out"].shape[0]):

            result = {
                "pos": out_dict["pos_out"][sample_idx],
                "ang": out_dict["ang_out"][sample_idx],
                "width": out_dict["width_out"][sample_idx],
                "cos": out_dict["cos_out"][sample_idx],
                "sin": out_dict["sin_out"][sample_idx]
            }
            results.append(result)
        return results

    @torch.no_grad()
    def compute_epistemic_uncertainty(self, batched_inputs=None, modalities=None, num_mc_samples=1):
        """ returns epistemic uncertainty per output
        dict keys: pos, cos, sin, width
        """
        if modalities is None:
            assert batched_inputs is not None, "Provide at least one of: [batched_inputs, modalities]!"
            modalities = self.preprocess_image(batched_inputs)

        # store current train state, set model to eval mode and enable dropout
        is_training = self.training
        self.eval()
        self.enable_dropout()

        # for each modality: replicate input according to num_mc_samples
        input_shape = modalities[0].tensor.shape

        assert input_shape[0]*num_mc_samples <= 160, (
        "Batch is too large. There is an issue with feeding large batches in a BN, see https://github.com/pytorch/pytorch/issues/32564!")

        for mod in modalities:
            if mod.tensor.shape[0] == 1:
                # using Tensor.expand() no additional memory is allocated
                # maybe this is a problem in case of multiple gpus
                mod.tensor = mod.tensor.expand(num_mc_samples*mod.tensor.shape[0], -1, -1, -1)
            else:
                # .expand() only works when the dim equals 1
                mod.tensor = mod.tensor.repeat(num_mc_samples, 1, 1, 1)
                mod.tensor = mod.tensor.contiguous()

        # apply mean and variance model and stack outputs
        out_dict = self.apply_network(modalities)
        out_dict = {k: out_dict[k] for k in out_dict.keys() if "out" in k}

        epistemic_uncertainties = {output_name: torch.zeros(input_shape, device=out_dict[output_name].device) for output_name in out_dict.keys()}

        # compute epistemic uncertainty for each sample
        for ouput_name in out_dict.keys():
            for sample_idx in range(input_shape[0]):
                mc_sampels = out_dict[ouput_name][sample_idx::input_shape[0]]
                epistemic_uncertainties[ouput_name][sample_idx] = torch.var(mc_sampels, dim=0)
        # reset model to train mode
        if is_training:
            self.train()
        elif not is_training:
            # disable dropout again when the model is in eval mode
            self.disable_dropout()

        return epistemic_uncertainties

    def enable_dropout(self):
        for module in self.modules():
            if "Dropout" in type(module).__name__:
                module.train()

    def disable_dropout(self):
        for module in self.modules():
            if "Dropout" in type(module).__name__:
                module.eval()
