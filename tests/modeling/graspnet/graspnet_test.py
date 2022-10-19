import pytest
import torch

from detectron2.config import get_cfg

from nicr_detectron2.modeling.meta_arch.grasp_net import GraspNet
from nicr_detectron2.config.default_config import *

@pytest.fixture
def graspnet_config():
    cfg = get_cfg()
    cfg = add_grasp_config(cfg)

    cfg.INPUT.MODALITIES = ['depth']
    cfg.INPUT.FORMAT = ['PGM']
    cfg.MODEL.PIXEL_MEANS = [[0]]
    cfg.MODEL.PIXEL_STDS = [[1]]

    return cfg

@pytest.mark.parametrize(['masked', 'expected_loss'],
                         [[False, {'pos_loss': 1, 'cos_loss': 1, 'sin_loss': 0, 'width_loss': 0}],
                          [True, {'pos_loss': 1, 'cos_loss': 0, 'sin_loss': 0, 'width_loss': 0}]])
def test_graspnet_loss(graspnet_config, masked, expected_loss):

    graspnet_config.MODEL.GRASP_NET.MASK_LOSS = masked

    model = GraspNet(graspnet_config)
    # model.pixel_mean_color.device = torch.device('cpu')

    img_shape = (100, 100)

    teacher = {
        'pos_gt': torch.zeros(img_shape),
        'cos_gt': torch.ones(img_shape),
        'sin_gt': torch.zeros(img_shape),
        'width_gt': torch.zeros(img_shape)
    }

    pos_output = torch.ones(img_shape)
    cos_output = torch.zeros(img_shape)
    sin_output = torch.zeros(img_shape)
    width_output = torch.zeros(img_shape)

    loss = model.compute_losses([teacher],
                                {'pos_out': pos_output,
                                 'cos_out': cos_output,
                                 'sin_out': sin_output,
                                 'width_out': width_output})

    for key, value in loss.items():
        assert torch.sum(value) == expected_loss[key]

