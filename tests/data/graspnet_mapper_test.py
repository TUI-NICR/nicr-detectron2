import pytest

from nicr_detectron2.data.graspnet_mapper import GraspDatasetMapper

@pytest.mark.parametrize(['rotate'], [[True], [False]])
def test_graspnet_mapper(rotate, grasp_depth_config, grasp_sample_dict):
    grasp_depth_config.INPUT.ROTATE = rotate
    mapper = GraspDatasetMapper(grasp_depth_config, is_train=False)

    sample = mapper(grasp_sample_dict)

    assert 'image' in sample
    for mod in grasp_depth_config.INPUT.MODALITIES:
        shape = sample['image'][mod].shape
        assert shape[1] == sample['height']
        assert shape[2] == sample['width']