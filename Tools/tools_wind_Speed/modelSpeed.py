import torch
import torch.nn as nn
from torchvision import models


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error (RMSE) loss function.

    This loss function calculates the RMSE between the predicted values and
    the true values.

    Args:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.

    Returns:
        torch.Tensor: The RMSE loss.

    Example:
        loss_fn = RMSELoss()
        loss = loss_fn(y_pred, y_true)
    """

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.view_as(y_pred)
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


# Define the main model class
class PretrainedWindModel(nn.Module):
    """
    PretrainedWindModel class represents a PyTorch module for a pretrained
    wind model.

    Args:
        hparams (dict): A dictionary containing hyperparameters for the model.

    Attributes:
        learning_rate (float): The learning rate for the model.
        hidden_size (int): The size of the hidden layer.
        dropout (float): The dropout rate.
        max_epochs (int): The maximum number of training epochs.
        num_workers (int): The number of workers for data loading.
        batch_size (int): The batch size for training.
        x_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The target training data.
        x_val (torch.Tensor): The input validation data.
        y_val (torch.Tensor): The target validation data.
        num_outputs (int): The number of output predictions.

    Methods:
        prepare_model(): Helper method to prepare the ResNet model.
        forward(image): Forward method for the model.
    """

    def __init__(self, hparams):
        super(PretrainedWindModel, self).__init__()
        self.hparams = hparams

        # Hyperparameters
        self.learning_rate = self.hparams.get("lr", 2e-4)
        self.hidden_size = self.hparams.get("embedding_dim", 50)
        self.dropout = self.hparams.get("dropout", 0.1)
        self.max_epochs = self.hparams.get("max_epochs", 1)
        self.num_workers = self.hparams.get("num_workers", 0)
        self.batch_size = self.hparams.get("batch_size", 10)
        self.x_train = self.hparams.get("x_train")
        self.y_train = self.hparams.get("y_train")
        self.x_val = self.hparams.get("x_val")
        self.y_val = self.hparams.get("y_val")
        self.num_outputs = 1  # One prediction only for regression

        # Start the model
        self.model = self.prepare_model()

    def prepare_model(self):
        """
        Helper method to prepare the ResNet model.

        Returns:
            torch.nn.Module: The prepared ResNet model.
        """
        res_model = models.resnet152(pretrained=True)
        res_model.fc = nn.Sequential(
            nn.Linear(2048, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_outputs),
        )
        return res_model

    def forward(self, image):
        """
        Forward method for the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The output prediction.
        """
        return self.model(image)
