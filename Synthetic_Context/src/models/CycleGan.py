import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchsummary import summary
import itertools
import pytorch_lightning as pl
from monai.networks.nets.unet import UNet
import torchmetrics

from src.models import PatchDiscriminator, ResnetGenerator
from src.models.utils import ImagePool, init_weights, set_requires_grad


class CycleGan(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # generator pair
        self.genX = UNet(spatial_dims=3,
                         in_channels=1,
                         out_channels=1,
                         channels=(4, 8, 16, 32, 64, 128, 256),
                         strides=(2, 2, 2, 2, 2, 2),
                         num_res_units = 3
                        )

        
        self.genY = UNet(spatial_dims=3,
                         in_channels=1,
                         out_channels=1,
                         channels=(4, 8, 16, 32, 64, 128, 256),
                         strides=(2, 2, 2, 2, 2, 2),
                         num_res_units = 3
                        )
        # self.genX = ResnetGenerator.get_generator()
        # self.genY = ResnetGenerator.get_generator()

        # discriminator pair
        self.disX = PatchDiscriminator.get_model()
        self.disY = PatchDiscriminator.get_model()

        self.lm = 10.0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None
        self.automatic_optimization = False
        self.accuracyDisX_train = torchmetrics.classification.Accuracy(task='binary')
        self.accuracyDisY_train = torchmetrics.classification.Accuracy(task='binary')
        self.accuracyDisX_test = torchmetrics.classification.Accuracy(task='binary')
        self.accuracyDisY_test = torchmetrics.classification.Accuracy(task='binary')
        self.accuracyDisX_val = torchmetrics.classification.Accuracy(task='binary')
        self.accuracyDisY_val = torchmetrics.classification.Accuracy(task='binary')

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)

    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999),
        )

        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999),
        )
        # gamma = lambda epoch: 1 - max(0, epoch + 1 - 5) / 101
        gamma = lambda epoch: 0.95 ** epoch
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
        According to the CycleGan paper, label for
        real is one and fake is zero.
        """
        if label.lower() == "real":
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)

        return F.mse_loss(predictions, target)

    def generator_step(self, imgA, imgB):
        """cycle images - using only generator nets"""
        fakeB = F.sigmoid(self.genX(imgA))*255
        cycledA = F.sigmoid(self.genY(fakeB))*255

        fakeA = F.sigmoid(self.genY(imgB))*255
        cycledB = F.sigmoid(self.genX(fakeA))*255

        sameB = F.sigmoid(self.genX(imgB))*255
        sameA = F.sigmoid(self.genY(imgA))*255

        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, "real")

        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, "real")

        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)

        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)

        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss

        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()

        return self.genLoss

    def discriminator_step(self, imgA, imgB):
        """Update Discriminator"""
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)

        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, "real")

        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, "fake")

        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, "real")

        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, "fake")

        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)

        return self.disLoss, predRealA, predFakeA, predRealB, predFakeB

    def training_step(self, batch, batch_idx):
        imgA, imgB = batch["A"], batch["B"]
        optimizer_g, optimizer_d = self.optimizers()
        
        
        set_requires_grad([self.disX, self.disY], False)
        self.toggle_optimizer(optimizer_g)
        
        genLoss = self.generator_step(imgA, imgB)
        self.log_dict({"train/gen_loss": genLoss.item()}, on_step=True, on_epoch=True)
        
        self.manual_backward(genLoss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        if self.global_step % 2 == 0:
            set_requires_grad([self.disX, self.disY], True)
            self.toggle_optimizer(optimizer_d)
            disLoss, predRealA, predFakeA, predRealB, predFakeB = self.discriminator_step(imgA, imgB)
        
            self.accuracyDisX_train.update(predRealA, torch.ones_like(predRealA))
            self.accuracyDisX_train.update(predFakeA, torch.zeros_like(predFakeA))
            
            self.accuracyDisY_train.update(predRealB, torch.ones_like(predRealB))
            self.accuracyDisY_train.update(predFakeB, torch.zeros_like(predFakeB))
            
            self.log_dict({"train/dis_loss": disLoss.item()}, on_step=True, on_epoch=True)
            self.manual_backward(disLoss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
        

    def on_train_epoch_end(self):
        self.log("train/accuracy_disX", self.accuracyDisX_train.compute(), on_step=False, on_epoch=True)
        self.log("train/accuracy_disY", self.accuracyDisY_train.compute(), on_step=False, on_epoch=True)
        self.accuracyDisX_train.reset()
        self.accuracyDisY_train.reset()

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def validation_step(self, batch, batch_idx):
        imgA, imgB = batch["A"], batch["B"]
        set_requires_grad([self.disX, self.disY], False)

        genLoss = self.generator_step(imgA, imgB)
        self.log_dict({"val/gen_loss": genLoss.item()}, on_step=False, on_epoch=True)
        disLoss, predRealA, predFakeA, predRealB, predFakeB = self.discriminator_step(imgA, imgB)
        self.accuracyDisX_val.update(predRealA, torch.ones_like(predRealA))
        self.accuracyDisX_val.update(predFakeA, torch.zeros_like(predFakeA))
        
        self.accuracyDisY_val.update(predRealB, torch.ones_like(predRealB))
        self.accuracyDisY_val.update(predFakeB, torch.zeros_like(predFakeB))
        
        self.log_dict({"val/dis_loss": disLoss.item()}, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        self.log("val/accuracy_disX", self.accuracyDisX_val.compute(), on_step=False, on_epoch=True)
        self.log("val/accuracy_disY", self.accuracyDisY_val.compute(), on_step=False, on_epoch=True)
        self.accuracyDisX_val.reset()
        self.accuracyDisY_val.reset()

    def test_step(self, batch, batch_idx):
        imgA, imgB = batch["A"], batch["B"]
        set_requires_grad([self.genX, self.genY, self.disX, self.disY], False)

        genLoss = self.generator_step(imgA, imgB)
        self.log_dict({"test/gen_loss": genLoss.item()}, on_step=False, on_epoch=True)
        disLoss, predRealA, predFakeA, predRealB, predFakeB = self.discriminator_step(imgA, imgB)
        self.accuracyDisX_test.update(predRealA, torch.ones_like(predRealA))
        self.accuracyDisX_test.update(predFakeA, torch.zeros_like(predFakeA))
        
        self.accuracyDisY_test.update(predRealB, torch.ones_like(predRealB))
        self.accuracyDisY_test.update(predFakeB, torch.zeros_like(predFakeB))
        
        self.log_dict({"test/dis_loss": disLoss.item()}, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        self.log("test/accuracy_disX", self.accuracyDisX_test.compute(), on_step=False, on_epoch=True)
        self.log("test/accuracy_disY", self.accuracyDisY_test.compute(), on_step=False, on_epoch=True)
        self.accuracyDisX_test.reset()
        self.accuracyDisY_test.reset()


if __name__ == "__main__":
    model = CycleGan()
