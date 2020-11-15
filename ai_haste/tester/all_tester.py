import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


def calculate_ssim(im1, im2, data_range=255, multichannel=True):
    if multichannel:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=True, full=True)[1]
        out_ssim = full_ssim.mean()
    else:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=False, full=True)[1]
        out_ssim = full_ssim.mean()

    return out_ssim


class AllChannelTester:
    def __init__(self, config, save_folder):
        self.config = config
        self.save_folder = save_folder
        self.c_out = 1
        self.kernel_size = 512
        self.stride = 256
        self.patch_batch_size = 8

    def load_models(self, models_dict):
        self.models = []
        for model_dict in models_dict:
            model = model_dict["model"].cuda()
            model = torch.nn.DataParallel(model)
            model.load_state_dict(
                torch.load(model_dict["model_path"])["model_state_dict"]
            )
            model = model.eval()
            self.models.append(model)

    def test(self, dataloader, models):
        self.load_models(models)
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(dataloader)):
                image_name = sample[1]
                preprocess_step = sample[2]
                preprocess_stats = sample[3]
                magnification = sample[4]

                input = sample[0].cuda().to(non_blocking=True)

                output_mask, output, mask_op_softmax = self.infer_full_image(
                    input, C_out,
                )

                if self.save_softmax:
                    np.save(
                        os.path.join(
                            self.softmax_save_folder, f"softmax_{image_name[0]}",
                        ),
                        mask_op_softmax.astype(np.float32),
                    )
                    np.save(
                        os.path.join(self.softmax_save_folder, f"mask_{image_name[0]}",),
                        mask.cpu()
                        .squeeze(0)
                        .numpy()
                        .transpose(1, 2, 0)
                        .astype(np.float32),
                    )

                intersection = torch.logical_and(mask, output_mask)
                union = torch.logical_or(mask, output_mask)
                iou = torch.sum(intersection) / torch.sum(union)
                ious.append(iou.item())

                output_8bit = (
                    (output[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )
                target_8bit = (
                    (target[0] * 255)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .astype("uint8")
                )

    def infer_full_image(self, input):
        B, C, W, H = input.shape
        pad_W = self.kernel_size - W % self.kernel_size
        pad_H = self.kernel_size - H % self.kernel_size

        patch_weights, _, _ = compute_pyramid_patch_weight_loss(
            self.kernel_size, self.kernel_size
        )

        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        _, W_pad, H_pad = input.shape
        patches = input.unfold(1, kernel_size, self.stride).unfold(
            2, self.kernel_size, self.stride
        )

        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, self.kernel_size, self.kernel_size)

        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        mask_op = []
        for batch_idx, sample1 in enumerate(dataloader):
            patch_mask_op, patch_op = self.models[0](sample1[0])
            patch_op = self.models[1](sample1[0])
            patch_op = self.models[2](sample1[0])
            op.append(patch_op)
            mask_op.append(patch_mask_op)
        op = torch.cat(op).permute(1, 0, 2, 3)

        op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        # weights = torch.ones_like(op)
        weights_op = (
            torch.from_numpy(patch_weights)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, self.c_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h)
        ).cuda()
        op = torch.mul(weights_op, op)
        op = F.fold(
            op,
            output_size=(W_pad, H_pad),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
        )
        weights_op = F.fold(
            weights_op,
            output_size=(W_pad, H_pad),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
        )
        op = torch.divide(op, weights_op)

        mask_op_softmax = (
            mask_op[:, :, :W, :H].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        )
        output = output[:, :, :W, :H]
        return output

    def write_output_images(
        self,
        output,
        target,
        output_mask,
        target_mask,
        image_name,
        preprocess_step,
        preprocess_stats,
        magnification,
    ):
        image_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)
        mask_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(mask_save_folder):
            os.makedirs(mask_save_folder)

        if preprocess_step == "normalize":
            min = preprocess_stats[0].cuda()
            max = preprocess_stats[1].cuda()
            output = (
                ((max - min) * output + min)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.uint16)
            )
            target = (
                ((max - min) * target + min)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.uint16)
            )
        elif preprocess_step == "standardize":
            mean = preprocess_stats[0].cuda()
            std = preprocess_stats[1].cuda()
            output = (
                ((output * std) + mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            )
            target = (
                ((target * std) + mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            )
        else:
            output = (output * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
            target = (target * 65535).cpu().numpy().transpose(1, 2, 0).astype(np.uint16)
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(image_save_folder, f"{filename}"), output[:, :, i],
            )
            cv2.imwrite(
                os.path.join(mask_save_folder, f"mask_{filename}"),
                (output_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )
        return output, target


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.
    :param width: Tile width
    :param height: Tile height
    :return: Since-channel image [Width x Height]
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De
