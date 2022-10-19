import pytest

from detectron2.config import get_cfg
from nicr_detectron2.config.default_config import add_multimodal_config, add_grasp_config

@pytest.fixture
def grasp_depth_config():
    cfg = get_cfg()
    cfg = add_grasp_config(cfg)

    cfg.INPUT.MODALITIES = ('depth',)
    cfg.INPUT.FORMAT = ['PGM']

    return cfg

@pytest.fixture
def multimodal_dataset_sample_dict():
    d = {
        'file_names': {'color': 'tests/example_data/00000.png', 'depth': 'tests/example_data/00000.tiff'}
    }

    return d


@pytest.fixture(scope='session')
def grasp_sample_dict():
    d = {
        'input_files': {'color_image': 'tests/example_data/00000.png', 'depth_image': 'tests/example_data/00000.tiff'},
        'label_files': {'quality': 'tests/example_data/00000_quality.npz', 'angle': 'tests/example_data/00000_angle.npz', 'width': 'tests/example_data/00000_width.npz'}
    }

    return d


@pytest.fixture
def multimodal_config():
    cfg = get_cfg()

    cfg = add_multimodal_config(cfg)

    cfg.INPUT.MODALITIES = ('color', 'depth')
    cfg.INPUT.FORMAT = ['BGR', 'PGM']
    cfg.INPUT.ROTATE = False

    return cfg
