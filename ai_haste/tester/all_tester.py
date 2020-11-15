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
        self.patch_batch_size = 1

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

                output_C1, output_C2, output_C3 = self.infer_all_channels1(input)

                self.write_output_images(
                    output[0],
                    image_name,
                    preprocess_step[0],
                    preprocess_stats,
                    magnification[0],
                )

    def infer_all_channels1(self, input):
        with torch.no_grad():
            B, C, W, H = input.shape
            pad_W = self.kernel_size - W % self.kernel_size
            pad_H = self.kernel_size - H % self.kernel_size

            input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
            _, W_padded, H_padded = input.shape
            patches = input.unfold(1, self.kernel_size, self.stride).unfold(
                2, self.kernel_size, self.stride
            )
            c, n_w, n_h, w, h = patches.shape
            patches = patches.contiguous().view(c, -1, self.kernel_size, self.kernel_size)

            fold = torch.nn.Fold(
                output_size=(W_padded, H_padded),
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.stride, self.stride),
            )

            patch_weights, _, _ = compute_pyramid_patch_weight_loss(
                self.kernel_size, self.kernel_size
            )
            weights_op = (
                torch.from_numpy(patch_weights)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(1, self.c_out, 1, n_w * n_h)
                .reshape(1, -1, n_w * n_h)
            ).cuda()

            dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
            batch_size = 1
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()

            torch.cuda.synchronize()
            with torch.cuda.stream(s1):
                c1_ops = []
                for batch_idx, sample_patch in enumerate(dataloader):
                    _, c1_op = self.models[0](sample_patch[0])
                    c1_ops.append(c1_op)

                c1_ops = torch.cat(c1_ops).permute(1, 0, 2, 3)
                c1_ops = c1_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
                c1_ops = torch.mul(weights_op, c1_ops)
                c1_ops = fold(c1_ops)

                weights_op = fold(weights_op)
                c1_ops = torch.divide(c1_ops, weights_op)
                c1_ops = c1_ops[:, :, :W, :H]

            with torch.cuda.stream(s2):
                c2_ops = []
                for batch_idx, sample_patch in enumerate(dataloader):
                    c2_op = self.models[1](sample_patch[0])
                    c2_ops.append(c2_op)

                c2_ops = torch.cat(c2_ops).permute(1, 0, 2, 3)
                c2_ops = c2_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
                c2_ops = torch.mul(weights_op, c2_ops)
                c2_ops = fold(c2_ops)

                weights_op = fold(weights_op)
                c2_ops = torch.divide(c2_ops, weights_op)
                c2_ops = c2_ops[:, :, :W, :H]

            with torch.cuda.stream(s3):
                c3_ops = []
                for batch_idx, sample_patch in enumerate(dataloader):
                    c3_op = self.models[2](sample_patch[0])
                    c3_ops.append(c3_op)
                c3_ops = torch.cat(c3_ops).permute(1, 0, 2, 3)
                c3_ops = c3_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
                c3_ops = torch.mul(weights_op, c3_ops)
                c3_ops = fold(c3_ops)

                weights_op = fold(weights_op)
                c3_ops = torch.divide(c3_ops, weights_op)
                c3_ops = c3_ops[:, :, :W, :H]

            torch.cuda.synchronize()
        return c1_ops, c2_ops, c3_ops

    def infer_all_channels(self, input):
        with torch.no_grad():
            B, C, W, H = input.shape
            pad_W = self.kernel_size - W % self.kernel_size
            pad_H = self.kernel_size - H % self.kernel_size

            input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
            _, W_padded, H_padded = input.shape
            patches = input.unfold(1, self.kernel_size, self.stride).unfold(
                2, self.kernel_size, self.stride
            )
            c, n_w, n_h, w, h = patches.shape
            patches = patches.contiguous().view(c, -1, self.kernel_size, self.kernel_size)

            fold = torch.nn.Fold(
                output_size=(W_padded, H_padded),
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.stride, self.stride),
            )

            patch_weights, _, _ = compute_pyramid_patch_weight_loss(
                self.kernel_size, self.kernel_size
            )
            weights_op = (
                torch.from_numpy(patch_weights)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(1, self.c_out, 1, n_w * n_h)
                .reshape(1, -1, n_w * n_h)
            ).cuda()

            dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
            batch_size = 1
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.patch_batch_size
            )

            c1_ops = []
            c2_ops = []
            c3_ops = []
            for batch_idx, sample_patch in enumerate(dataloader):
                _, c1_op = self.models[0](sample_patch[0])
                c2_op = self.models[1](sample_patch[0])
                c3_op = self.models[2](sample_patch[0])
                c1_ops.append(c1_op)
                c2_ops.append(c2_op)
                c3_ops.append(c3_op)
            c1_ops = torch.cat(c1_ops).permute(1, 0, 2, 3)
            c1_ops = c1_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)

            c2_ops = torch.cat(c2_ops).permute(1, 0, 2, 3)
            c2_ops = c2_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)

            c3_ops = torch.cat(c3_ops).permute(1, 0, 2, 3)
            c3_ops = c3_ops.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)

            c1_ops = torch.mul(weights_op, c1_ops)
            c2_ops = torch.mul(weights_op, c2_ops)
            c3_ops = torch.mul(weights_op, c3_ops)

            c1_ops = fold(c1_ops)
            c2_ops = fold(c2_ops)
            c3_ops = fold(c3_ops)

            weights_op = fold(weights_op)

            c1_ops = torch.divide(c1_ops, weights_op)
            c2_ops = torch.divide(c2_ops, weights_op)
            c3_ops = torch.divide(c3_ops, weights_op)

            c1_ops = c1_ops[:, :, :W, :H]
            c2_ops = c2_ops[:, :, :W, :H]
            c3_ops = c3_ops[:, :, :W, :H]

        return c1_ops, c2_ops, c3_ops

    def write_output_images(
        self, output, image_name, preprocess_step, preprocess_stats, magnification,
    ):
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
