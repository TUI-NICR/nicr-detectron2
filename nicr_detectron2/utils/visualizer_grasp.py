from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

import torch
import cv2

class GraspVisualizer:
    def __init__(self,
                 rgb_image: np.ndarray) -> None:
        self._rgb_image = rgb_image

    def visualize_prediction(self,
                             predictions: Dict[str, torch.Tensor],
                             labels: Optional[Dict[str, torch.Tensor]]):
        print(predictions.keys())
        f = plt.figure(figsize=(15, 5))
        axes = []

        input_grid_size = 2
        ncols = 5
        nrows = 2

        has_epistemic_uncertainty = False
        has_aleatoric_uncertainty = False

        if 'epistemic' in predictions:
            # if we estimate uncertainties make space for extra images
            # spec = f.add_gridspec(ncols=6, nrows=3)
            ncols += 1
            nrows += 1
            input_grid_size += 1
            has_epistemic_uncertainty = True
        if 'aleatoric' in predictions:
            has_aleatoric_uncertainty = True
            ncols += 1
            nrows += 1
            input_grid_size += 1

        spec = f.add_gridspec(ncols=ncols, nrows=nrows)

        ax = f.add_subplot(spec[:, :input_grid_size])
        ax.imshow(self._rgb_image)
        axes.append(ax)

        current_row = 0

        for i, key in enumerate(['pos', 'ang', 'width']):
            ax = f.add_subplot(spec[current_row, (input_grid_size+i)])

            if key == 'pos':
                ax.imshow(predictions[key].cpu().numpy().squeeze(),
                          vmin=0, vmax=1)
                ax.set_ylabel('prediction')
            else:
                ax.imshow(predictions[key].cpu().numpy().squeeze())

            axes.append(ax)

        current_row += 1

        if has_epistemic_uncertainty:
            uncertainties = predictions['epistemic']
            ax = f.add_subplot(spec[current_row, (input_grid_size)])
            ax.imshow(uncertainties['pos'].cpu().numpy().squeeze(),
                      vmin=0)
            axes.append(ax)

            ax.set_ylabel('epistemic')

            ax = f.add_subplot(spec[current_row, (input_grid_size+1)])
            cos_var = uncertainties['cos'].cpu().numpy().squeeze()
            sin_var = uncertainties['sin'].cpu().numpy().squeeze()

            ang_img = np.zeros((cos_var.shape[0], cos_var.shape[1], 3))
            ang_img[:, :, 0] = cos_var
            ang_img[:, :, 1] = sin_var

            # scale to [0, 255] with fixed maximum uncertainty
            # TODO: Find nice way of defining this maximum uncertainty
            ang_img /= ang_img.max()
            # ang_img *= 255

            ax.imshow(ang_img)
            axes.append(ax)

            ax = f.add_subplot(spec[current_row, (input_grid_size+2)])
            ax.imshow(uncertainties['width'].cpu().numpy().squeeze())
            axes.append(ax)

            current_row += 1

        if has_aleatoric_uncertainty:
            uncertainties = predictions['aleatoric']
            ax = f.add_subplot(spec[current_row, (input_grid_size)])
            ax.imshow(uncertainties['pos'].cpu().numpy().squeeze(),
                      vmin=0)
            axes.append(ax)

            ax.set_ylabel('aleatoric')

            ax = f.add_subplot(spec[current_row, (input_grid_size+1)])
            cos_var = uncertainties['cos'].cpu().numpy().squeeze()
            sin_var = uncertainties['sin'].cpu().numpy().squeeze()

            ang_img = np.zeros((cos_var.shape[0], cos_var.shape[1], 3))
            ang_img[:, :, 0] = cos_var
            ang_img[:, :, 1] = sin_var

            # scale to [0, 255] with fixed maximum uncertainty
            # TODO: Find nice way of defining this maximum uncertainty
            ang_img /= ang_img.max()
            # ang_img *= 255

            ax.imshow(ang_img)
            axes.append(ax)

            ax = f.add_subplot(spec[current_row, (input_grid_size+2)])
            ax.imshow(uncertainties['width'].cpu().numpy().squeeze())
            axes.append(ax)

            current_row += 1

        for i, key in enumerate(['pos_gt', 'angle_gt', 'width_gt']):
            ax = f.add_subplot(spec[current_row, (input_grid_size+i)])

            if i == 0:
                ax.set_ylabel('GT')

            ax.imshow(np.moveaxis(labels[key].cpu().numpy(), 0, -1))

            axes.append(ax)

        current_row += 1

        for ax in axes:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        #     ax.axis('off')
        f.tight_layout()

        return f, axes
