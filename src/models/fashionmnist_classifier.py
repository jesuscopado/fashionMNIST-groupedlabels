from typing import Optional, Tuple

import pytorch_lightning as pl
from torch import nn, optim, Tensor
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class FashionClassifier(pl.LightningModule):
    """FashionClassifier is a PyTorch Lightning module for classifying fashion items."""

    def __init__(self, num_classes: int = 5, class_weights: Optional[Tensor] = None):
        """
        Initializes the FashionClassifier with specified number of classes and optional class weights.

        Args:
            num_classes (int): The number of classes for classification.
            class_weights (Tensor, optional): A tensor of size num_classes to provide weights for the classes.
        """
        super(FashionClassifier, self).__init__()

        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='weighted')
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='weighted')
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the model.
        """
        return self.mobilenet(x)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer for the model.

        Returns:
            optim.Optimizer: The optimizer.
        """
        return optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step for each batch.

        Args:
            batch (Tuple[Tensor, Tensor]): Input batch of data and labels.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Update and log metrics
        self.log('train_loss', loss)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.train_precision(y_hat.softmax(dim=-1), y)
        self.train_recall(y_hat.softmax(dim=-1), y)
        self.train_f1(y_hat.softmax(dim=-1), y)

        return loss

    def on_train_epoch_end(self) -> None:
        """Operations to perform at the end of each training epoch."""
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('train_precision_epoch', self.train_precision.compute())
        self.log('train_recall_epoch', self.train_recall.compute())
        self.log('train_f1_epoch', self.train_f1.compute())

        # Reset metrics
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step for each batch.

        Args:
            batch (Tuple[Tensor, Tensor]): Input batch of data and labels.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Update and log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.val_precision(y_hat.softmax(dim=-1), y)
        self.val_recall(y_hat.softmax(dim=-1), y)
        self.val_f1(y_hat.softmax(dim=-1), y)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Operations to perform at the end of each validation epoch."""
        self.log('val_acc_epoch', self.val_acc.compute(), prog_bar=True)
        self.log('val_precision_epoch', self.val_precision.compute(), prog_bar=True)
        self.log('val_recall_epoch', self.val_recall.compute(), prog_bar=True)
        self.log('val_f1_epoch', self.val_f1.compute(), prog_bar=True)

        # Reset metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Test step for each batch.

        Args:
            batch (Tuple[Tensor, Tensor]): Input batch of data and labels.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # Update and log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.test_precision(y_hat.softmax(dim=-1), y)
        self.test_recall(y_hat.softmax(dim=-1), y)
        self.test_f1(y_hat.softmax(dim=-1), y)

        return loss

    def on_test_epoch_end(self) -> None:
        """Operations to perform at the end of each test epoch."""
        self.log('test_acc_epoch', self.test_acc.compute(), prog_bar=True)
        self.log('test_precision_epoch', self.test_precision.compute(), prog_bar=True)
        self.log('test_recall_epoch', self.test_recall.compute(), prog_bar=True)
        self.log('test_f1_epoch', self.test_f1.compute(), prog_bar=True)

        # Reset metrics
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
