import pytest
from detectron2.data import DatasetCatalog

from nicr_detectron2.engine.grasp_net_trainer import GraspNetTrainer
import nicr_detectron2.modeling # pylint: disable=unused-import

@pytest.fixture(scope='session')
def test_dataset(grasp_sample_dict):
    def _return_sample():
        return [grasp_sample_dict]

    DatasetCatalog.register('_train', _return_sample)

@pytest.mark.parametrize(['meta_arch'], [['GGCNN2'], ['GRConvNet']])
def test_grasp_net_trainer(grasp_depth_config, test_dataset, tmp_path, meta_arch):
    grasp_depth_config.SOLVER.MAX_ITER = 1
    grasp_depth_config.SOLVER.IMS_PER_BATCH = 1

    grasp_depth_config.DATASETS.TRAIN = ('_train',)
    grasp_depth_config.DATASETS.TEST = ('_train',)

    grasp_depth_config.OUTPUT_DIR = str(tmp_path)

    grasp_depth_config.MODEL.META_ARCHITECTURE = meta_arch

    grasp_depth_config.MODEL.PIXEL_MEANS = [[0]]
    grasp_depth_config.MODEL.PIXEL_STDS = [[1]]

    grasp_depth_config.INPUT.RANDOM_FLIP = 'none'
    grasp_depth_config.INPUT.MAX_SIZE_TRAIN = 320

    trainer = GraspNetTrainer(grasp_depth_config)

    trainer.train()
