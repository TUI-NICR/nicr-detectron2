import copy
import numpy as np

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import (
    resize, rotate, center_crop, crop
)

class GraspTransforms():
    def __init__(self,
                shape=(320, 320),
                random_flip=False,
                random_zoom=False,
                random_rotate=False,
                grasp_center_crop=False):

        self.shape = shape
        self.random_flip = random_flip
        self.random_zoom = random_zoom
        self.random_rotate = random_rotate
        self.grasp_center_crop = grasp_center_crop

    def __call__(self, images, labels, grasp_list=None):
        flip, zoom, rotation = self.get_params()

        transformed_images = copy.deepcopy(images)
        transformed_labels = copy.deepcopy(labels)

        # crop around the center of labeled grasps
        if self.grasp_center_crop:
            assert grasp_list is not None
            self.apply_grasp_center_crop(transformed_images, transformed_labels, grasp_list)
        else:
            # crop to shortest edge
            shape = None
            shortest_edge = 0
            for key in transformed_images.keys():
                if shape is None:
                    shape = transformed_images[key].shape[1:]
                    shortest_edge = min(shape)
                transformed_images[key] = center_crop(transformed_images[key], (shortest_edge, shortest_edge))
            for key in transformed_labels.keys():
                transformed_labels[key] = center_crop(transformed_labels[key], (shortest_edge, shortest_edge))

        # transform images
        for key in transformed_images.keys():
            # resize
            transformed_images[key] = resize(transformed_images[key], self.shape, interpolation=InterpolationMode.NEAREST)
            if self.random_rotate:
                transformed_images[key] = rotate(transformed_images[key], np.degrees(rotation))
            if self.random_zoom:
                transformed_images[key] = self.center_zoom(transformed_images[key], zoom)
            if self.random_flip:
                raise NotImplementedError

        # transform labels
        for key in transformed_labels.keys():
            # resize
            transformed_labels[key] = resize(transformed_labels[key], self.shape, interpolation=InterpolationMode.NEAREST)
            if self.random_rotate:
                transformed_labels[key] = rotate(transformed_labels[key], np.degrees(rotation))
            if self.random_zoom:
                transformed_labels[key] = self.center_zoom(transformed_labels[key], zoom)
            if self.random_flip:
                raise NotImplementedError

        # special treatment of the angle image
        label_mask = transformed_labels["quality"] != 0.0
        if self.random_rotate:
            transformed_labels["angle"][label_mask] += rotation
            transformed_labels["angle"][label_mask] = (transformed_labels["angle"][label_mask] + np.pi/2) % np.pi - np.pi/2
        if self.random_flip:
            raise NotImplementedError

        return transformed_images, transformed_labels, (flip, zoom, rotation)

    def apply_grasp_center_crop(self, transformed_images, transformed_labels, grasp_list):
        # get all grasp points an concat them along axis 0
        grasp_points = [grasp.points for grasp in grasp_list.grasps]
        grasp_points = np.concatenate(grasp_points, axis=0)
        # bounding box of all grasp points
        min = np.min(grasp_points, axis=0)
        max = np.max(grasp_points, axis=0)
        # add 20% of image width as margin around the grasps
        # clip to image boundaries
        max_limit = np.array(next(iter(transformed_images.values())).shape[1:][::-1])
        min = np.floor(np.clip(min - 0.2 * max_limit, 0, max_limit)).astype(int)
        max = np.ceil(np.clip(max + 0.2 * max_limit, 0, max_limit)).astype(int)
        # width an height
        delta = max-min
        # now crop labels and images
        # grasp points come as [x,y] --> delta[0]=width, delta[1]=height, min[1]=y0, min[0]=x0
        for key in transformed_images.keys():
            transformed_images[key] = crop(transformed_images[key], min[1], min[0], delta[1], delta[0])
        for key in transformed_labels.keys():
            transformed_labels[key] = crop(transformed_labels[key], min[1], min[0], delta[1], delta[0])

    def center_zoom(self, image, zoom):
        c, h, w = image.shape
        crop_shape = (int(zoom*h),int(zoom*w))
        cropped_image = center_crop(image, crop_shape)
        zoomed_image = resize(cropped_image, (h,w), interpolation=InterpolationMode.NEAREST)
        return zoomed_image

    def get_params(self):
        zoom = 1.0
        if self.random_zoom:
            zoom = np.random.uniform(0.5, 1.0)

        rotation = 0.0
        if self.random_rotate:
            rotations = [0, np.pi/2, np.pi, 3*np.pi/2]
            rotation = np.random.choice(rotations)

        flip = False
        if self.random_flip:
            flip = np.random.choice([True, False])

        return flip, zoom, rotation