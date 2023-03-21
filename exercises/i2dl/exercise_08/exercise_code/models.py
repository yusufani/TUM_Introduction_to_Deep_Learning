import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        # Hint: You can use nn.Sequential() to define your encoder.            #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        # Hint: wrap them up in nn.Sequential().                               #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        ########################################################################

        #self.encoder = torch.nn.Sequential(
        #    nn.Linear(self.input_size, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, latent_dim)
        #)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, latent_dim),
            nn.LeakyReLU(),

        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        #self.decoder = torch.nn.Sequential(
        #    nn.Linear(latent_dim, latent_dim * 2),
        #    nn.LeakyReLU(),
        #    nn.Linear(latent_dim * 2, latent_dim * 4),
        #    nn.ReLU(),
        #    nn.Linear(latent_dim * 4, latent_dim * 8),
        #    nn.BatchNorm1d(latent_dim * 8),
        #    nn.ReLU(),
        #    nn.Linear(latent_dim * 8, latent_dim * 16),
        #    nn.ReLU(),
        #    nn.Dropout(),
        #    nn.Linear(latent_dim * 16, output_size)
        #)


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):
        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hparams['lr'])
                                          #,weight_decay=self.hparams['weight_decay'])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similraly to the way it is shown in      #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################

        self.encoder.train()  # Set the model to training mode
        self.decoder.train()

        self.optimizer.zero_grad()  # Reset the gradients - VERY important! Otherwise they accumulate.
        batch = batch.to(self.device)
        batch = batch.view(batch.shape[0], -1)
        reconstruction = self.forward(batch)
        loss = loss_func(reconstruction, batch)  # Compute the loss over the predictions and the ground truth.
        loss.backward()  # Stage 2: Backward().
        self.optimizer.step()  # Stage 3: Update the parameters.

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        self.decoder.eval()
        self.encoder.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            batch = batch.view(batch.shape[0], -1)
            reconstruction = self.forward(batch)
            loss = loss_func(reconstruction, batch)  # Compute the loss over the predictions and the ground truth.
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):
        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################

        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.hparams["hidden_size"]),
            nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(self.hparams["hidden_size"], self.hparams["hidden_size"]//2),
            nn.LeakyReLU(),
            nn.Linear(self.hparams["hidden_size"]//2, self.hparams["num_classes"])
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer = torch.optim.Adam(self.parameters(), self.hparams["lr"])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
