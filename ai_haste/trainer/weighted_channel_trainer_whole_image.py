import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

## Basically Ebba playing around with weighted_channel_trainer
class WeightedChannelTrainerForMasks:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.lr_scheduler_config = config["lr_scheduler"]
        self.save_folder = save_folder
        self.save_interval = config["save_interval"]
        self.image_folder = os.path.join(self.save_folder, "images")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            self.models.append(torch.nn.DataParallel(model))

        self.model = self.models[0]
        self.optimizer_config["args"]["params"] = self.model.parameters()
        optimizer = getattr(optim, self.optimizer_config["type"])(
            **self.optimizer_config["args"]
        )
        if self.lr_scheduler_config:
            self.lr_scheduler_config["args"]["optimizer"] = optimizer
            self.lr_scheduler = getattr(
                optim.lr_scheduler, self.lr_scheduler_config["type"]
            )(**self.lr_scheduler_config["args"])

        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_loss = np.inf

        for epoch in range(1, self.epochs):
            train_losses = self._train_epoch(
                train_dataloader, optimizer, criterions, epoch
            )
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_train.to_csv(
                    os.path.join(self.save_folder, "losses_train.csv"), index=False
                )

            if epoch % 5 == 0:
                valid_losses, epoch_loss = self._valid_epoch(
                    valid_dataloader, criterions, epoch
                )
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_valid.to_csv(
                    os.path.join(self.save_folder, "losses_valid.csv"), index=False
                )
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "epoch_loss": epoch_loss,
                        },
                        os.path.join(self.save_folder, f"model_best_loss.pth"),
                    )

            if self.lr_scheduler:
                self.lr_scheduler.step()

        return self.save_folder

    def infer_full_image(self, input, C_out, kernel_size=256, stride=128):
        self.model.eval()
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size
        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        _, W_pad, H_pad = input.shape
        patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)
        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        op = []
        mask_op = []
        for batch_idx, sample1 in enumerate(dataloader):
            patch_mask_op, patch_op = self.model(sample1[0])
            op.append(patch_op)
            mask_op.append(patch_mask_op)
        op = torch.cat(op).permute(1, 0, 2, 3)
        mask_op = torch.cat(mask_op).permute(1, 0, 2, 3)
        op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        mask_op = mask_op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        weights = torch.ones_like(op)
        op = F.fold(
            op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        mask_op = F.fold(
            mask_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        weights = F.fold(
            weights,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        op = torch.divide(op, weights)
        mask_op = torch.divide(mask_op, weights)
        # op = op.view(C_out, n_w, n_h, w, h)
        # mask_op = mask_op.view(C_mask_out, n_w, n_h, w, h)
        # output_h = n_w * w
        # output_w = n_h * h
        # op = op.permute(0, 1, 3, 2, 4).contiguous()
        # mask_op = mask_op.permute(0, 1, 3, 2, 4).contiguous()
        # op = op.view(C_out, output_h, output_w)
        # mask_op = mask_op.view(C_mask_out, output_h, output_w)
        output = torch.clamp(op, 0.0, 1.0)
        mask_op = mask_op.argmax(dim=1).unsqueeze(1)
        output = output[:, :, :W, :H]
        mask_output = mask_op[:, :, :W, :H]
        return mask_output, output

    def _train_epoch(self, dataloader, optimizer, criterions, epoch):
        self.model.train()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]
        total_losses = [0.0 for i in range(len(criterions))]
        log_interval = 2
        loss_log = [[] for i in range(len(criterions))]
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1][:,0].unsqueeze(1).cuda().to(non_blocking=True)
            mask = sample[1][:,1].cuda().to(non_blocking=True)

            optimizer.zero_grad()

            output = self.model(input)
            output = torch.clamp(output, 0.0, 1.0)
            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                if  'Mask' in criterion["loss"].__class__.__name__:
                    loss_class = criterion["loss"](output, target, mask)
                else:
                    loss_class = criterion["loss"](output, target)

                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                loss_desc = f"Train Epoch {epoch} Current Total Loss:{running_total_loss/log_interval:.5f}"
                for i, criterion in enumerate(criterions):
                    loss_name = criterion["loss"].__class__.__name__
                    loss_desc += (
                        f" Current {loss_name} {running_losses[i]/log_interval:.5f}"
                    )
                running_loss_desc.set_description_str(loss_desc)
                #running_losses[0] = 0.0
                #running_losses[1] = 0.0
                running_total_loss = 0.0
        if epoch % self.save_interval == 0:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )

        loss_desc = f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        for i, criterion in enumerate(criterions):
            loss_name = criterion["loss"].__class__.__name__
            loss_desc += f" Total {loss_name} {total_losses[i]/len(dataloader):.5f}"
        total_loss_desc.set_description_str(loss_desc)

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model.eval()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        log_interval = 1
        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                image_name = sample[2]
                preprocess_step = sample[3]
                preprocess_stats = sample[4]

                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)
                C_out = target.shape[1]

                output = self.infer_full_image(input, C_out)

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                if epoch % 10 == 0 and batch_idx == 0:
                    self.write_output_images(
                        output[0],
                        target[0],
                        image_name,
                        preprocess_step[0],
                        preprocess_stats,
                        epoch,
                    )

                outer.update(1)

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        )
        return loss_log, total_loss / len(dataloader)


def write_output_images(output, target, image_name, preprocess_step, preprocess_stats):
    if preprocess_step == "normalize":
        min = preprocess_stats[0].cuda()
        max = preprocess_stats[1].cuda()
        output1 = (
            ((max - min) * output + min)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint16)
        )
        target1 = (
            ((max - min) * target + min)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint16)
        )
    elif preprocess_step == "standardize":
        mean
    y = 1

