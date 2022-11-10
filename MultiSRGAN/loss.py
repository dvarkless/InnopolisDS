import torch
from torch import nn
from torchvision.models.resnet import ResNet50_Weights, resnet50


class FullLoss(nn.Module):
    def __init__(self, coeff_image=1, coeff_adv=0.01, coeff_perc=0.07):
        super().__init__()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        assert 0 < coeff_image <= 1
        assert 0 < coeff_adv <= 1
        assert 0 < coeff_perc <= 1
        self.coeff_image = coeff_image
        self.coeff_adv = coeff_adv
        self.coeff_perc = coeff_perc
        self.clear_losses()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.log(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        self.image_loss = self.image_loss + self.coeff_image * image_loss
        self.adv_loss = self.adv_loss + self.coeff_adv * adversarial_loss
        self.perc_loss = self.perc_loss + self.coeff_perc * perception_loss

        return self.coeff_image * image_loss \
            + self.coeff_adv * adversarial_loss \
            + self.coeff_perc * perception_loss

    def get_losses(self):
        return self.image_loss, self.adv_loss, self.perc_loss

    def clear_losses(self):
        self.image_loss = 0.0
        self.adv_loss = 0.0
        self.perc_loss = 0.0


class MSELoss(nn.Module):
    def __init__(self, coeff_image=1, coeff_adv=0, coeff_perc=0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        assert 0 < coeff_image <= 1
        assert 0 < coeff_adv <= 1
        assert 0 < coeff_perc <= 1
        self.coeff_image = coeff_image
        self.coeff_adv = coeff_adv
        self.coeff_perc = coeff_perc
        self.clear_losses()

    def forward(self, out_images, target_images):
        image_loss = self.mse_loss(out_images, target_images)
        self.image_loss = self.image_loss + self.coeff_image * image_loss
        return image_loss

    def get_losses(self):
        return self.coeff_image * self.image_loss, 0, 0

    def clear_losses(self):
        self.image_loss = 0.0


class AdversarialLoss(nn.Module):
    def __init__(self, coeff_image=0, coeff_adv=1, coeff_perc=0.2):
        super().__init__()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        assert 0 < coeff_image <= 1
        assert 0 < coeff_adv <= 1
        assert 0 < coeff_perc <= 1
        self.coeff_image = coeff_image
        self.coeff_adv = coeff_adv
        self.coeff_perc = coeff_perc
        self.clear_losses()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.log(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        self.adv_loss = self.adv_loss + self.coeff_adv * adversarial_loss
        self.perc_loss = self.perc_loss + self.coeff_perc * perception_loss

        return self.coeff_adv * adversarial_loss \
            + self.coeff_perc * perception_loss

    def get_losses(self):
        return 0, self.adv_loss, self.perc_loss

    def clear_losses(self):
        self.adv_loss = 0.0
        self.perc_loss = 0.0


if __name__ == "__main__":
    g_loss = FullLoss()
    print(g_loss)
