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
from torch import nn
from collections import deque


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


def disc_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class raLSGAN_segTrainer:
    def __init__(self, config, save_folder):
        self.config = config
        self.epochs = config["epochs"]
        self.gen_optimizer_config = config["gen_optimizer"]
        self.disc_optimizer_config = config["disc_optimizer"]
        self.gen_lr_scheduler_config = config["gen_lr_scheduler"]
        self.disc_lr_scheduler_config = config["disc_lr_scheduler"]
        self.save_folder = save_folder
        self.save_interval = config["save_interval"]
        self.image_folder = os.path.join(self.save_folder, "images")

        self.pretrain = config["pretrain"]
        self.n_pretrain_epochs = config["n_pretrain_epochs"]
        self.adv_loss_weight = config["adv_loss_weight"]

        self.load_pretrained = config["load_pretrained"]
        self.pretrained_model_path = config["pretrained_model_path"]
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def train(self, train_dataloader, valid_dataloader, models, criterions):
        self.models = []
        for model in models:
            model = model.cuda()
            self.models.append(torch.nn.DataParallel(model))

        self.generator = self.models[0]
        self.discriminator = self.models[1]

        if self.load_pretrained:
            checkpoint = torch.load(self.pretrained_model_path)
            self.generator.load_state_dict(checkpoint["model_state_dict"])

        self.fake_pool = ImagePool(50)
        self.real_pool = ImagePool(50)
        self.grad_norm = 1.0

        self.gen_optimizer_config["args"]["params"] = self.generator.parameters()
        optimizer_G = getattr(optim, self.gen_optimizer_config["type"])(
            **self.gen_optimizer_config["args"]
        )
        self.disc_optimizer_config["args"]["params"] = self.discriminator.parameters()
        optimizer_D = getattr(optim, self.disc_optimizer_config["type"])(
            **self.disc_optimizer_config["args"]
        )
        if self.gen_lr_scheduler_config:
            self.gen_lr_scheduler_config["args"]["optimizer"] = optimizer_G
            self.gen_lr_scheduler = getattr(
                optim.lr_scheduler, self.gen_lr_scheduler_config["type"]
            )(**self.gen_lr_scheduler_config["args"])

        if self.disc_lr_scheduler_config:
            self.disc_lr_scheduler_config["args"]["optimizer"] = optimizer_D
            self.disc_lr_scheduler = getattr(
                optim.lr_scheduler, self.disc_lr_scheduler_config["type"]
            )(**self.disc_lr_scheduler_config["args"])

        all_train_losses_log = [[] for i in range(len(criterions) + 1)]
        all_valid_losses_log = [[] for i in range(len(criterions))]
        best_loss = np.inf

        if self.pretrain:
            print("Pretraining  for ", self.n_pretrain_epochs, " epochs \n")
            for pre_epoch in range(1, self.n_pretrain_epochs + 1):
                train_losses = self._pretrain_epoch(
                    train_dataloader, optimizer_G, criterions, pre_epoch
                )
        print("Training for real!")
        for epoch in range(1, self.epochs + 1):

            train_losses = self._train_epoch(
                train_dataloader, optimizer_G, optimizer_D, criterions, epoch
            )

            for i in range(len(all_train_losses_log)):
                all_train_losses_log[i].extend(train_losses[i])

                df_train = pd.DataFrame(all_train_losses_log).transpose()
                loss_names = [x["loss"].__class__.__name__ for x in criterions]
                loss_names.insert(0, "GANLoss")

                df_train.columns = loss_names
                df_train.to_csv(
                    os.path.join(self.save_folder, "losses_train.csv"), index=False
                )

            if epoch % 10 == 0:
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
                            "model_state_dict": self.generator.state_dict(),
                            "epoch": epoch,
                            "epoch_loss": epoch_loss,
                        },
                        os.path.join(self.save_folder, f"model_best_loss.pth"),
                    )

            if self.gen_lr_scheduler:
                self.gen_lr_scheduler.step()
            if self.disc_lr_scheduler:
                self.disc_lr_scheduler.step()

        return self.save_folder

    def _train_epoch(self, dataloader, optimizer_G, optimizer_D, criterions, epoch):
        self.generator.train()
        self.discriminator.train()

        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions) + 1)]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions) + 1)]

        log_interval = 2
        loss_log = [[] for i in range(len(criterions) + 1)]

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            mask = sample[1][:, 0].cuda().to(non_blocking=True)
            mask_onehot = (
                F.one_hot(mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
            )
            target = sample[1][:, 1].unsqueeze(1).cuda().to(non_blocking=True)

            ## ----------------------
            ## Train Discriminator
            ## ----------------------
            optimizer_D.zero_grad()

            with torch.no_grad():
                _, fake = self.generator(input.float())
                # if 'NonWeightedBCELogitsLoss' in [x["loss"].__class__.__name__ for x in criterions]:
                #     fake = nn.Sigmoid()(fake)

            pred_fake = self.discriminator(torch.cat([input, fake.detach()], 1))

            self.fake_pool.add(pred_fake)

            pred_real = self.discriminator(torch.cat([input, target.float()], 1))

            self.real_pool.add(pred_real)

            loss_D = (
                torch.mean((pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2)
                + torch.mean((pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)
            ) / 2

            loss_D.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.grad_norm
            )

            optimizer_D.step()

            ## ------------------
            ##  Train Generators
            ## ------------------

            optimizer_G.zero_grad()

            mask_op, output = self.generator(input.float())
            # output = torch.clamp(output, 0.0, 1.0)

            pred_fake = self.discriminator(torch.cat([input, output], 1))
            pred_real = self.discriminator(torch.cat([input, target.float()], 1))

            # Adversarial loss

            loss_GAN = (
                torch.mean((pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2)
                + torch.mean((pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)
            ) / 2

            losses = []
            loss = 0

            running_losses[0] += loss_GAN.item()
            total_losses[0] += loss_GAN.item()
            loss_log[0].append(loss_GAN.item())
            losses.append(loss_GAN.item())
            loss += self.adv_loss_weight * loss_GAN

            loss_segmentation = criterions[0]["loss"](mask_op, mask_onehot.float())
            running_losses[1] += loss_segmentation.item()
            total_losses[1] += loss_segmentation.item()
            loss_log[1].append(loss_segmentation.item())
            losses.append(loss_segmentation.item())
            loss += criterions[0]["weight"] * loss_segmentation

            for i, criterion in enumerate(criterions[1:]):
                loss_class = criterion["loss"](output, target)
                running_losses[i + 2] += loss_class.item()
                total_losses[i + 2] += loss_class.item()
                loss_log[i + 2].append(loss_class.item())
                losses.append(loss_class.item())
                loss += criterion["weight"] * loss_class

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_norm)

            optimizer_G.step()

            total_loss += loss.item()
            running_total_loss += loss.item()

            outer.update(1)
            if (batch_idx + 1) % log_interval == 0:
                loss_desc = f"Train Epoch {epoch} Current Total Loss:{running_total_loss/log_interval:.5f}"

                loss_name = "GANLoss"
                loss_desc += f" Current {loss_name} {running_losses[0]/log_interval:.5f}"
                running_losses[0] = 0.0
                for i, criterion in enumerate(criterions):
                    loss_name = criterion["loss"].__class__.__name__
                    loss_desc += (
                        f" Current {loss_name} {running_losses[i +1]/log_interval:.5f}"
                    )
                    running_losses[i + 1] = 0.0
                running_loss_desc.set_description_str(loss_desc)
                running_total_loss = 0.0

        loss_desc = f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"

        loss_name = "GANLoss"
        loss_desc += f" Total {loss_name} {total_losses[0]/len(dataloader):.5f}"

        for i, criterion in enumerate(criterions):
            loss_name = criterion["loss"].__class__.__name__
            loss_desc += f" Total {loss_name} {total_losses[i + 1]/len(dataloader):.5f}"
        total_loss_desc.set_description_str(loss_desc)

        if epoch % self.save_interval == 0:
            torch.save(
                self.generator.state_dict(),
                os.path.join(self.save_folder, f"model_epoch_{epoch}.pth"),
            )

        return loss_log

    def _valid_epoch(self, dataloader, criterions, epoch):
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
                mask = sample[1][:, 0].unsqueeze(1).cuda().to(non_blocking=True)
                mask_onehot = F.one_hot(mask.long()).squeeze(1).permute(0, 3, 1, 2)
                C_mask_out = mask_onehot.shape[1]
                target = sample[1][:, 1].unsqueeze(1).cuda().to(non_blocking=True)
                C_out = target.shape[1]

                output_mask, output = self.infer_full_image(
                    input, C_out, C_mask_out, kernel_size=256, stride=128
                )

                # if 'NonWeightedBCELogitsLoss' in [x["loss"].__class__.__name__ for x in criterions]:
                #     output = nn.Sigmoid()(output)

                losses = []
                loss = 0

                loss_log[0].append(loss_segmentation.item())
                losses.append(loss_segmentation.item())
                loss += loss_segmentation

                for i, criterion in enumerate(criterions[1:]):
                    loss_class = criterion["loss"](output, target)
                    loss_log[i + 1].append(loss_class.item())
                    losses.append(loss_class.item())
                    loss += criterion["weight"] * loss_class

                total_loss += loss.item()

                if epoch % 10 == 0 and batch_idx % 5 == 0:
                    self.write_output_images(
                        output[0],
                        target[0],
                        output_mask[0],
                        mask[0],
                        image_name,
                        preprocess_step[0],
                        preprocess_stats,
                        magnification[0],
                        epoch,
                    )

                outer.update(1)

        total_loss_desc.set_description_str(
            f"Valid Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        )
        return loss_log, total_loss / len(dataloader)

    def infer_full_image(self, input, C_out, C_mask_out, kernel_size=256, stride=128):
        self.generator.eval()
        B, C, W, H = input.shape
        pad_W = kernel_size - W % kernel_size
        pad_H = kernel_size - H % kernel_size

        x, _, _ = compute_pyramid_patch_weight_loss(kernel_size, kernel_size)

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
            patch_mask_op, patch_op = self.generator(sample1[0])
            op.append(patch_op)
            mask_op.append(patch_mask_op)
        op = torch.cat(op).permute(1, 0, 2, 3)
        mask_op = torch.cat(mask_op).permute(1, 0, 2, 3)

        op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        mask_op = mask_op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
        # weights = torch.ones_like(op)
        weights_op = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, C_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h)
        ).cuda()
        op = torch.mul(weights_op, op)
        op = F.fold(
            op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        weights_op = F.fold(
            weights_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        op = torch.divide(op, weights_op)

        weights_mask_op = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, C_mask_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h)
        ).cuda()
        mask_op = torch.mul(weights_mask_op, mask_op)
        mask_op = F.fold(
            mask_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        weights_mask_op = F.fold(
            weights_mask_op,
            output_size=(W_pad, H_pad),
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
        )
        mask_op = torch.divide(mask_op, weights_mask_op)
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
        epoch,
    ):
        image_save_folder = os.path.join(self.image_folder, f"{magnification}_images")
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)

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
                os.path.join(image_save_folder, f"epoch_{epoch}_output_{filename}"),
                output[:, :, i],
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_target_{filename}"),
                target[:, :, i],
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_output_mask_{filename}"),
                (output_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )
            cv2.imwrite(
                os.path.join(image_save_folder, f"epoch_{epoch}_target_mask_{filename}"),
                (target_mask.cpu().numpy().transpose(1, 2, 0)[:, :, i] * 65535).astype(
                    np.uint16
                ),
            )

    def _pretrain_epoch(self, dataloader, optimizer, criterions, epoch):
        self.generator.train()
        outer = tqdm(total=len(dataloader), desc="Batches Processed:", position=0)
        running_loss_desc = tqdm(total=0, position=1, bar_format="{desc}")
        total_loss_desc = tqdm(total=0, position=2, bar_format="{desc}")
        total_loss = 0.0
        total_losses = [0.0 for i in range(len(criterions))]

        running_total_loss = 0.0
        running_losses = [0.0 for i in range(len(criterions))]

        log_interval = 2
        loss_log = [[] for i in range(len(criterions))]

        num_batches = 0
        batches_processed = 0

        for batch_idx, sample in enumerate(dataloader):
            input = sample[0].cuda().to(non_blocking=True)
            mask = sample[1][:, 0].cuda().to(non_blocking=True)
            mask_onehot = (
                F.one_hot(mask.unsqueeze(1).long()).squeeze(1).permute(0, 3, 1, 2)
            )
            target = sample[1][:, 1].unsqueeze(1).cuda().to(non_blocking=True)

            mask_op, output = self.generator(input)
            # output = torch.clamp(output, 0.0, 1.0)
            losses = []
            loss = 0

            loss_segmentation = criterions[0]["loss"](mask_op, mask_onehot.float())
            running_losses[0] += loss_segmentation.item()
            total_losses[0] += loss_segmentation.item()
            loss_log[0].append(loss_segmentation.item())
            losses.append(loss_segmentation.item())
            loss += criterions[0]["weight"] * loss_segmentation

            for i, criterion in enumerate(criterions[1:]):
                loss_class = criterion["loss"](output, target)
                running_losses[i + 1] += loss_class.item()
                total_losses[i + 1] += loss_class.item()
                loss_log[i + 1].append(loss_class.item())
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
                    running_losses[i] = 0.0
                running_loss_desc.set_description_str(loss_desc)
                running_total_loss = 0.0

        loss_desc = f"Train Epoch {epoch} Total Loss:{total_loss/len(dataloader):.5f}"
        for i, criterion in enumerate(criterions):
            loss_name = criterion["loss"].__class__.__name__
            loss_desc += f" Total {loss_name} {total_losses[i]/len(dataloader):.5f}"
        total_loss_desc.set_description_str(loss_desc)

        return loss_log


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
