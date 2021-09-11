from model import DigitRecogniser
from dataset import MNISTDataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    max_epochs = 10
    steps_per_epoch = 470
    model = DigitRecogniser(
        lr=1e-0, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch
    )
    data = MNISTDataModule(batch_size=128)

    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
    trainer.fit(model, data)
