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

def plot_multi(cols, rows, images, **kwargs):
    fig, axs = plt.subplots(rows, cols)
    axs = axs.ravel()

    for ax, im in zip(axs, images):
        ax.imshow(im, **kwargs)

    plt.show(block=True)

def to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)

class PureSegTrainer:
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

        #self.criterion_seg = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 2.0]).cuda())#torch.nn.BCEWithLogitsLoss()
        self.criterion_seg = torch.nn.CrossEntropyLoss()
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

        #criterion_seg = torch.nn.CrossEntropyLoss()

        

        self.model.train()

        for batch_idx, sample in enumerate(dataloader):
            

            #plot_multi(3,3,[sample[1][i] for i in range(8)])

            # plt.imshow(sample[0][0][0])
            # plt.imshow(sample[1][0], alpha = 0.4)
            # plt.show()
            input = sample[0].cuda().to(non_blocking=True)
            target = sample[1].cuda().to(non_blocking=True)
            #target = to_one_hot(target.long(), 2).permute(0,3,1,2).cuda().to(non_blocking=True)

            optimizer.zero_grad()

            output_seg = self.model(input)
        
            loss_seg = self.criterion_seg(output_seg, target.long())

            running_seg_loss += loss_seg.item()
            total_seg_loss += loss_seg.item()
            

            loss_seg.backward()
            optimizer.step()

            total_loss += loss_seg.item()
            running_total_loss += loss_seg.item()

            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                loss_desc = f"Train Epoch {epoch} Current Total Loss:{running_total_loss/log_interval:.5f}"
                loss_desc += (
                        f" Current {'Seg_loss'} {running_seg_loss/log_interval:.5f}"
                    )
                running_loss_desc.set_description_str(loss_desc)
                running_seg_loss = 0.0

        #plot_multi(2,1,[output_seg[0][i].detach().cpu().numpy() for i in range(2)])
        if epoch % self.save_interval == 0:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )


        total_loss_desc.set_description_str(
            f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader)} Current Seg Loss: {total_seg_loss/len(dataloader)}"
        )
        return loss_log


    def _valid_epoch(self, dataloader, criterions, epoch):
        self.model.eval()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        #total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        log_interval = 1
        loss_log = [[] for i in range(len(criterions))]
        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                image_name = sample[2]
                # preprocess_step = sample[3]
                # preprocess_stats = sample[4]

                input = sample[0].cuda().to(non_blocking=True)
                target = sample[1]
                #mask = to_one_hot(target.long(), 2).permute(0,3,1,2).cuda().to(non_blocking=True)

                C_out = 1

                output_seg = self.infer_full_image(input, C_out)

                #total_loss += self.criterion_seg(output_seg, mask.float()).item()

                if epoch % 10 == 0 and batch_idx == 0:
                    self.write_output_images(
                        output_seg[0][0],
                        target[0],
                        image_name,
                        epoch,
                    )

                outer.update(1)
        
        # total_loss_desc.set_description_str(
        #     f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        # )
        #print('Percentage of non zero el.: ', torch.count_nonzerooutput_seg.detach().cpu())/output_seg.detach().cpu().numel(), ' %')
        return loss_log, total_loss / len(dataloader)

    def infer_full_image(self, input, C_out, kernel_size=256, stride=256):
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size

        input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)
        patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)

        c, n_w, n_h, w, h = patches.shape
        patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)

        dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        seg = []


        for batch_idx, sample1 in enumerate(dataloader):
            seg_out = self.model(sample1[0])

            seg.append(seg_out)
        seg = torch.cat(seg).permute(1, 0, 2, 3)

        output_h = n_w * w
        output_w = n_h * h

        #seg = torch.argmax(torch.nn.Sigmoid()(seg), dim=0).unsqueeze(0)
        seg = torch.argmax(torch.nn.Softmax(dim=0)(seg), dim=0).unsqueeze(0)

        seg = seg.view(C_out, n_w, n_h, w, h)
        seg = seg.permute(0, 1, 3, 2, 4).contiguous()
        seg = seg.view(C_out, output_h, output_w)

        output_seg = seg[:, :W, :H].unsqueeze(0)
        return output_seg

    def write_output_images(
        self, output_seg, mask, image_name, epoch):
        for i, filename in enumerate(image_name):
            cv2.imwrite(
                os.path.join(self.image_folder, f"epoch_{epoch}_target_mask_{filename}.tif"),
                        (mask.detach().cpu().numpy()*255).astype(np.uint8))

            cv2.imwrite(
                os.path.join(self.image_folder, f"epoch_{epoch}_output_seg_{filename}.tif"),
                        (output_seg.detach().cpu().numpy()*255).astype(np.uint8))