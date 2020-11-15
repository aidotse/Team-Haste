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
import matplotlib.pyplot as plt

def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)

class DualTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.lr_scheduler_config = config["lr_scheduler"]
        self.save_folder = save_folder
        # if config["save_name"] != "":
        #     self.save_folder = self.save_folder + "_" + config["save_name"]
        # if not os.path.exists(self.save_folder):
        #     os.makedirs(self.save_folder)
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
        optimizer = getattr(optim, self.optimizer_config["type"])(**self.optimizer_config["args"])
        if self.lr_scheduler_config:
            self.lr_scheduler_config["args"]["optimizer"] = optimizer
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_config["type"])(
                **self.lr_scheduler_config["args"]
            )

        # criterion = criterions[0]
        all_train_losses_log = [[] for i in range(len(criterions))]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_loss = np.inf

        for epoch in range(1, self.epochs):
            train_losses = self._train_epoch(train_dataloader, optimizer, criterions, epoch)
            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                df_train.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_train.to_csv(os.path.join(self.save_folder, "losses_train.csv"), index=False)

            if epoch % 5 == 0:
                valid_losses, epoch_loss = self._valid_epoch(valid_dataloader, criterions, epoch)
                for i in range(len(all_valid_losses_log)):
                    all_valid_losses_log[i].extend(valid_losses[i])

                df_valid = pd.DataFrame(all_valid_losses_log).transpose()
                df_valid.columns = [x["loss"].__class__.__name__ for x in criterions]
                df_valid.to_csv(os.path.join(self.save_folder, "losses_valid.csv"), index=False)
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
        # return self.model

    def _train_epoch(self, dataloader, optimizer, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]
        total_losses = [0.0 for i in range(len(criterions))]
        log_interval = 2
        loss_log = [[] for i in range(len(criterions))]

        running_seg_loss = 0.0
        total_seg_loss = 0.0

        criterion_seg = torch.nn.CrossEntropyLoss()

        self.model.train()

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1][:,0].unsqueeze(1).cuda().to(non_blocking=True)
            mask = sample[1][:,1].cuda().to(non_blocking=True)

            optimizer.zero_grad()

            output, output_seg = self.model(input)
            
            output = torch.clamp(output, 0.0, 1.0)

            losses = []
            loss = 0
            for i, criterion in enumerate(criterions):
                loss_class = criterion["loss"](output, target)
                running_losses[i] += loss_class.item()
                total_losses[i] += loss_class.item()
                loss_log[i].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss_seg = criterion_seg(output_seg, mask.long())

            running_seg_loss += loss_seg.item()
            total_seg_loss += loss_seg.item()
            loss = loss + loss_seg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                running_loss_desc.set_description_str(
                    f"Train Epoch {epoch} Current Total Loss:{running_total_loss/log_interval} Current L1 Loss: {running_losses[0]/log_interval} Current SSIM Loss: {running_losses[1]/log_interval} Current Seg Loss: {running_seg_loss/log_interval}"
                )
                running_losses[0] = 0.0
                running_losses[1] = 0.0
                running_seg_loss = 0.0
                running_total_loss = 0.0
        if epoch % self.save_interval == 0:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )
        # print(
        #     f"Training Epoch:{epoch}\
        #             Current loss: {total_loss/len(dataloader)}"
        # )
        total_loss_desc.set_description_str(
            f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader)} Current L1 Loss: {total_losses[0]/len(dataloader)} Current SSIM Loss: {total_losses[1]/len(dataloader)} Current Seg Loss: {total_seg_loss/len(dataloader)}"
        )
        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        # running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        log_interval = 1
        loss_log = [[] for i in range(len(criterions))]

        self.model.eval()

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                B, C, W, H = input.shape
                kernel_size = 512
                stride = 512
                patches = input.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

                target = sample[1][:,0].unsqueeze(1).cuda().to(non_blocking=True)
                mask = sample[1][:,1].cuda().to(non_blocking=True)

                C_out = target.shape[1]

                op_tensor = torch.zeros(B, C_out, W, H).cuda()
                seg_tensor = torch.zeros(B, 2, W, H).cuda()
                for i in range(patches.shape[2]):
                    for j in range(patches.shape[3]):
                        op, seg = self.model(patches[:, :, i, j])
                        op_tensor[
                            :, :, i * stride : (i + 1) * stride, j * stride : (j + 1) * stride,
                        ] = op

                        seg_tensor[
                            :, :, i * stride : (i + 1) * stride, j * stride : (j + 1) * stride,
                        ] = seg

                output = torch.clamp(op_tensor, 0.0, 1.0)

                seg_preds = torch.argmax(torch.nn.Softmax(dim=1)(seg_tensor), dim=1)

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                if epoch % 10 == 0:
                    if batch_idx == 0:
                        if output.shape[1] < 3:
                            cv2.imwrite(
                                os.path.join(self.image_folder, f"target_epoch_{epoch}.tif"),
                                (target[0][0].detach().cpu().numpy() * 255).astype(np.uint8),
                            )
                            cv2.imwrite(
                                os.path.join(self.image_folder, f"mask_epoch_{epoch}.tif"),
                                (seg_preds[0].detach().cpu().numpy()*255).astype(np.uint8),
                            )
                            cv2.imwrite(
                                os.path.join(self.image_folder, f"target_mask_epoch_{epoch}.tif"),
                                (mask[0].detach().cpu().numpy()*255).astype(np.uint8),
                            )

                            cv2.imwrite(
                                os.path.join(self.image_folder, f"output_epoch_{epoch}.tif"),
                                (output[0][0].detach().cpu().numpy() * 255).astype(np.uint8),
                            )
                        else:
                            cv2.imwrite(
                                os.path.join(self.image_folder, f"target_epoch_{epoch}.tif"),
                                (target[0].detach().cpu().numpy() * 255)
                                .astype(np.uint8)
                                .transpose(1, 2, 0),
                            )

                            cv2.imwrite(
                                os.path.join(self.image_folder, f"mask_epoch_{epoch}.tif"),
                                (seg_preds[0].detach().cpu().numpy()*255).astype(np.uint8),
                            )

                            cv2.imwrite(
                                os.path.join(self.image_folder, f"target_mask_epoch_{epoch}.tif"),
                                (mask[0].detach().cpu().numpy()*255).astype(np.uint8),
                            )

                            cv2.imwrite(
                                os.path.join(self.image_folder, f"output_epoch_{epoch}.tif"),
                                (output[0].detach().cpu().numpy() * 255)
                                .astype(np.uint8)
                                .transpose(1, 2, 0),
                            )
                outer.update(1)
        # print(
        #     f"Validation Epoch:{epoch}\
        #             Current loss: {total_loss/len(dataloader)}"
        # )
        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader)}"
        )
        return loss_log, total_loss / len(dataloader)

    def _valid_epoch_1(self, dataloader, criterions, epoch):
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss = 0.0
        log_interval = 1
        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1].cuda().to(non_blocking=True)

                outputs = []
                for i in range(input.shape[1]):
                    output = self.model(input[:, i])
                    output = torch.clamp(output, 0.0, 1.0)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=0)

                if len(target.shape) == 4:
                    target = target.permute(1, 0, 2, 3)
                else:
                    target = target[0]

                losses = []
                loss = 0
                for i, criterion in enumerate(criterions):
                    loss_class = criterion["loss"](outputs, target)
                    loss_log[i].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                outer.update(1)
        print(
            f"Validation Epoch:{epoch}\
                    Current loss: {total_loss/len(dataloader)}"
        )
        return loss_log, total_loss / len(dataloader)
