import argparse

import pytorch_lightning as pl
import torch
from torchvision import transforms

from src.datamodule.fashionmnist_datamodule import FashionMNISTDataModule
from src.models.fashionmnist_classifier import FashionClassifier


def main(args):
    # Initialize the data module
    fashion_mnist_dm = FashionMNISTDataModule(
        batch_size=args.batch_size,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
    )

    # Call setup to prepare the data
    fashion_mnist_dm.setup_manual()

    # Initialize model
    device = torch.device("cuda:0" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    class_weights = fashion_mnist_dm.class_weights.to(device)
    model = FashionClassifier(class_weights=class_weights)

    # Load the model weights
    model.load_from_checkpoint(args.checkpoint_path, map_location=device, strict=False)
    model.to(device)

    # Test the model
    pl.Trainer(
        accelerator=args.accelerator,
    ).test(
        model=model,
        datamodule=fashion_mnist_dm
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FashionMNIST Classifier')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint to evaluate')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'], help='Type of accelerator to use')

    args = parser.parse_args()

    main(args)
