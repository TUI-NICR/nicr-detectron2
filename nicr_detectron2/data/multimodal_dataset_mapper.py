
import copy
import logging
from PIL import Image
from detectron2.data.transforms.augmentation_impl import RandomRotation
import numpy as np
from typing import List, Optional, Tuple, Union
import torch
import re
import cv2
import time

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.transform import ResizeTransform

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]
logger = logging.getLogger(__name__)


class DatasetMapper:
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
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        modalities: Tuple = ()
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.modalities = modalities

    @classmethod
    def build_augmentation(self, cfg, is_train):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        if is_train and cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        return augmentation

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = cls.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "modalities": cfg.INPUT.MODALITIES
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret
    
    def _load_img(self, path, format):
        # load image based on its format
        # PGM is handled by using cv2
        # t = time.time()
        if format in ['PGM']:
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            img = np.expand_dims(img, -1)
            # print('d', time.time() - t)
            return img
        else:
            img = utils.read_image(path, format=format)
            # print('c', time.time() - t)
            return img

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        
        

        images = {mod: self._load_img(dataset_dict["file_names"][mod], format=frmt) for mod, frmt in zip(self.modalities, self.image_format)}
        [utils.check_image_size(dataset_dict, image) for _, image in images.items()]

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        sem_seg_gt, transforms = self.process_images(images, sem_seg_gt)

        image_shape = images[list(images.keys())[0]].shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = {mod: torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))) for mod, image in images.items()}
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    def process_images(self, images, sem_seg_gt):
        transforms = None
        image_keys = list(images.keys())
        first_color_key = None

        # look for a color image
        # we use it to compute transforms
        for key in image_keys:
            if key.lower().startswith('color'):
                first_color_key = key
        
        if first_color_key is not None:
            # if we have a color image use it for augmentation
            # to get transforms
            image = images[first_color_key]
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)

            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

            images[first_color_key] = image

        for key, format in zip(self.modalities, self.image_format):
            if key == first_color_key:
                continue

            image = images[key]

            if transforms is None:
                logger.warn('[DatasetMapper] No color image found. If transforms contain resize they might be applied incorrectly (interpolation method: BILINEAR)')
                aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
                transforms = self.augmentations(aug_input)
                image, sem_seg_gt = aug_input.image, aug_input.sem_seg
            else:
                for trans in transforms:
                    if isinstance(trans, ResizeTransform) and format == 'PGM' or isinstance(trans, RandomRotation) and format == 'PGM':
                        # if resize and pgm image use nearest interpolation
                        image = trans.apply_image(image, interp=Image.NEAREST)
                    else:
                        image = trans.apply_image(image)
                    if len(image.shape) < 3:
                        image = np.expand_dims(image, -1)
            images[key] = image
        return sem_seg_gt, transforms
