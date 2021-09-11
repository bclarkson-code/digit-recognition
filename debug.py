from dataset import MNISTDataModule

if __name__ == "__main__":
    data = MNISTDataModule()
    data.setup()
    loader = data.val_dataloader()
    print(next(iter(loader)))
