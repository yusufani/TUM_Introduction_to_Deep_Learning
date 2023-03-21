"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        #https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html#torchvision.models.EfficientNet_B4_Weights

        encoder = models.alexnet(weights=True, progress=True).eval()
        for param in encoder.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            *(list(encoder.children())[:-1]),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(64, 32, 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Upsample(size=30, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 120, 120)
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (32, 240, 240)
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

        )
        #print(self.model)




        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = x.to(self.device)
        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
