import os

import copy
import logging
import numpy as np
import scipy.sparse as sparse
from typing import List, Optional, Tuple, Union
import torch
import cv2
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from nicr_grasping.datatypes.grasp import RectangleGraspList

from .transforms.graspnet_transform import GraspTransforms

__all__ = ["GraspDatasetMapper"]
logger = logging.getLogger(__name__)

GLOBAL_IMAGE_CACHE = {}

# TODO: Maybe inherit from MultiModalDatasetMapper?


class GraspDatasetMapper():
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is based on the default DatasetMapper of detectron.
    It can handle multiple input images of different types.
    Used modalities (keys of file_name in datasetdict) can be specified in cfg.INPUT.MODALITIES
    together with their format with cfg.INPUT.FORMAT (which is now a list)

    The callable currently does the following:

    1. Read the images from "file_name" which is now a dict
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        size: int,
        augmentations: GraspTransforms,
        image_format: str,
        modalities: Tuple = ()
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            size: input/output of shape (size x size)
            augmentations: a list of augmentations or deterministic transforms to apply
        """
        self.size = size
        self.is_train = is_train
        self.image_format = image_format
        self.augmentations = augmentations
        self.modalities = modalities

        self.cache = {}

        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def build_augmentations(cls, cfg, is_train):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """
        if is_train:
            size = cfg.INPUT.MAX_SIZE_TRAIN
        else:
            size = cfg.INPUT.MAX_SIZE_TEST

        random_flip = is_train and cfg.INPUT.RANDOM_FLIP != "none"
        random_rotate = is_train and cfg.INPUT.ROTATE
        random_zoom = is_train and cfg.INPUT.ZOOM
        return GraspTransforms(shape=(size, size),
                               random_flip=random_flip,
                               random_zoom=random_zoom,
                               random_rotate=random_rotate)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augmentations = cls.build_augmentations(cfg, is_train=is_train)

        if is_train:
            size = cfg.INPUT.MAX_SIZE_TRAIN
        else:
            size = cfg.INPUT.MAX_SIZE_TEST

        ret = {
            "is_train": is_train,
            "size": size,
            "augmentations": augmentations,
            "image_format": cfg.INPUT.FORMAT,
            "modalities": cfg.INPUT.MODALITIES
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # get the sample id
        # this info is usefull for the GraspNetPredictionSaver
        sample_id = int(list(dataset_dict['input_files'].values())[0].split('/')[-1].split('.')[0])

        # load images
        images = {mod: self._load_img(dataset_dict["input_files"][mod+"_image"], format=frmt) for mod, frmt in zip(self.modalities, self.image_format)}
        for _, image in images.items():
            utils.check_image_size(dataset_dict, image)

        if not self.is_train and "color" not in self.modalities:
            if "color_image" not in dataset_dict["input_files"].keys():
                print("No color image provided!")
                images["color"] = copy.deepcopy(np.repeat(images["depth"][:, :, np.newaxis], 3, axis=2))
            else:
                images["color"] = self._load_img(dataset_dict["input_files"]["color_image"], format='RGB')

        # get input original image and image shape for evaluation
        # original_images = copy.deepcopy(images)
        if "color" in images.keys():
            original_shape = images["color"].shape[:-1]
        elif "depth" in images.keys():
            original_shape = images["depth"].shape
        else:
            raise NotImplementedError

        # images as torch.tensor and dim should be CxHxW with C=1 or C=3
        for key in images.keys():
            if len(images[key].shape) == 2:
                images[key] = np.expand_dims(images[key], axis=0)
            images[key] = torch.from_numpy(images[key].astype(np.float32)).permute(2, 0, 1)

        # load label files
        labels = {key: sparse.load_npz(dataset_dict["label_files"][key]) for key in dataset_dict["label_files"].keys()}
        for key in labels.keys():
            label_np = labels[key].todense()
            if len(label_np.shape)==2:
                label_np = np.expand_dims(label_np, axis=0)
            labels[key] = torch.from_numpy(label_np.astype(np.float32))

        # WIP: load GraspRectangleList
        sample_name = os.path.basename(dataset_dict["label_files"]["quality"].replace("_quality.npz", ".pkl"))
        grasp_rectangles_path = os.path.dirname(
                os.path.dirname(dataset_dict["label_files"]["quality"])
            )
        grasp_rectangles_path = os.path.join(grasp_rectangles_path, f"grasp_lists/{sample_name}")
        if os.path.isfile(grasp_rectangles_path) and not self.is_train:
            grasp_rectangles = RectangleGraspList.load_from_file(grasp_rectangles_path)
        else:
            if not self.is_train:
                print("Invalid rectangle path!")
            grasp_rectangles = RectangleGraspList()

        # apply data augmentation
        images, labels, _ = self.augmentations(images, labels, grasp_rectangles)

        # compute sin and cos gt from ang img and expand dims
        labels["cos"] = torch.cos(2*labels["angle"])
        labels["sin"] = torch.sin(2*labels["angle"])

        # scale width
        # TODO: find a better way
        labels["width"] = np.clip(labels["width"], 0.0, 150.0)/150.0

        # norm depth
        # TODO: 0 in depth image should be treated specially as those represent no measurement
        if "depth" in images.keys():
            images["depth"] = (images["depth"]-images["depth"].min()) / (images["depth"].max()+1e-7)

        # norm color
        if "color" in images.keys():
            images["color"] /= 255.0

        # output dict
        # TODO: width and heigth should contain the same info as original_shape
        # TODO: return modified dict and dont create a new one
        return {
            "image": {key: images[key] for key in images.keys()},
            "pos_gt": labels["quality"],
            "cos_gt": labels["cos"],
            "sin_gt": labels["sin"],
            "width_gt": labels["width"],
            "width": next(iter(images.values())).shape[2],
            "height": next(iter(images.values())).shape[1],
            "rgb_grasps": None,  # TODO: remove rgb grasps
            "grasp_rectangles": grasp_rectangles,
            "original_shape": original_shape,
            "angle_gt": labels["angle"],
            "sample_id": sample_id
        }

    def _load_img(self, path, format):
        # load image based on its format
        # PGM is handled by using cv2
        if format in ['PGM']:
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            img = np.expand_dims(img, -1)
        else:
            img = utils.read_image(path, format=format)

        return img


# if __name__ == '__main__':
#     from nicr_detectron2.data import graspnet_datasets
#     from detectron2.data import DatasetCatalog, MetadataCatalog
#     from torchvision.utils import save_image

#     train_dataset = DatasetCatalog.get("cornell_train")
#     test_dataset = DatasetCatalog.get("cornell_test")
#     print(f"length of train dataset: {len(train_dataset)} length of test dataset: {len(test_dataset)}")
#     sample = test_dataset[-1]

#     augmentations = GraspTransforms(shape=(320,320), random_flip=False, random_zoom=True, random_rotate=True, grasp_center_crop=True)
#     augmentations_test = GraspTransforms(shape=(320,320), random_flip=False, random_zoom=False, random_rotate=False, grasp_center_crop=True)
#     train_mapper = GraspDatasetMapper(is_train=True, size=320, augmentations=augmentations, modalities=("color","depth"), image_format=["RGB","PGM"])
#     test_mapper = GraspDatasetMapper(is_train=False, size=320, augmentations=augmentations_test, modalities=("color","depth"), image_format=["RGB","PGM"])
#     train_dict = train_mapper(copy.deepcopy(sample))
#     test_dict = test_mapper(copy.deepcopy(sample))

#     train_tensors = [
#         train_dict["image"]["color"],
#         train_dict["image"]["depth"][0].repeat(3,1,1),
#         train_dict["pos_gt"][0].repeat(3,1,1),
#         train_dict["cos_gt"][0].repeat(3,1,1),
#         train_dict["sin_gt"][0].repeat(3,1,1),
#         train_dict["width_gt"][0].repeat(3,1,1),
#         train_dict["angle_gt"][0].repeat(3,1,1)]

#     test_tensors = [
#         test_dict["image"]["color"],
#         test_dict["image"]["depth"][0].repeat(3,1,1),
#         test_dict["pos_gt"][0].repeat(3,1,1),
#         test_dict["cos_gt"][0].repeat(3,1,1),
#         test_dict["sin_gt"][0].repeat(3,1,1),
#         test_dict["width_gt"][0].repeat(3,1,1),
#         train_dict["angle_gt"][0].repeat(3,1,1)]

#     tensors = train_tensors + test_tensors
#     #save_image(train_dict["image"]["color"],"test.png")
#     save_image(tensors,"test.png",nrow=7,scale_each=True,normalize=True)
