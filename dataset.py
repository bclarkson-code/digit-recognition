from torchvision.datasets import MNIST
import pytorch_lightning as pl
import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=None, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        if num_workers is None:
            self.num_workers = 4
        else:
            self.num_workers = num_workers

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        # split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_train, [55000, 5000]
            )
        if stage == (None, "test"):
            self.mnist_test = MNIST(
                os.getcwd(), train=False, transform=transform
            )

    # return the dataloader for each split
    def train_dataloader(self):
        mnist_train = DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return mnist_test
