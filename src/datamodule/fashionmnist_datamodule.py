from collections import Counter
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class FashionMNISTDataModule(pl.LightningDataModule):
    """FashionMNISTDataModule is a PyTorch Lightning data module for handling Fashion MNIST dataset."""

    def __init__(self, batch_size: int = 32, transform: Optional[Any] = None):
        """Initializes the FashionMNISTDataModule with specified batch size and optional transform.

        Args:
            batch_size (int): Size of each data batch.
            transform (Callable, optional): A function/transform to preprocess the data.
        """
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform if transform is not None else transforms.ToTensor()

    def setup_manual(self) -> None:
        """Manually set up the data module. Downloads the data and prepares it for use."""
        self.trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=self.transform)

        label_mapping = {
            0: 0, 2: 0, 4: 0, 6: 0,  # Upper part
            1: 1,  # Bottom part
            3: 2,  # One piece
            5: 3, 7: 3, 9: 3,  # Footwear
            8: 4  # Bags
        }

        self.trainset.targets = [label_mapping[label] for _, label in self.trainset]
        self.testset.targets = [label_mapping[label] for _, label in self.testset]

        label_counts = Counter(self.trainset.targets)
        total_count = sum(label_counts.values())
        class_weights = {cls: 1.0 / (count / float(total_count)) for cls, count in label_counts.items()}
        self.class_weights = torch.FloatTensor([class_weights[i] for i in sorted(class_weights.keys())])

    def train_dataloader(self) -> DataLoader:
        """Create a DataLoader for the training set.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Create a DataLoader for the validation set.

        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.testset, batch_size=self.batch_size)  # TODO: create a proper val set

    def test_dataloader(self) -> DataLoader:
        """Create a DataLoader for the test set.

        Returns:
            DataLoader: DataLoader for the test set.
        """
        return DataLoader(self.testset, batch_size=self.batch_size)
