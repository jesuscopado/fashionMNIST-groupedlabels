import unittest

import torch
from torchvision import transforms

from src.datamodule.fashionmnist_datamodule import FashionMNISTDataModule
from src.models.fashionmnist_classifier import FashionClassifier


class TestTraining(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.num_classes = 5
        self.model = FashionClassifier(self.num_classes)
        self.data_module = FashionMNISTDataModule(
            batch_size=self.batch_size,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        )
        self.data_module.setup_manual()

    def test_forward_pass(self):
        x = torch.randn(self.batch_size, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))

    def test_training_step(self):
        batch = next(iter(self.data_module.train_dataloader()))
        loss = self.model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)


if __name__ == '__main__':
    unittest.main()
