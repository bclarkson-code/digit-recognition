from model import DigitRecogniser
from dataset import MNISTDataModule
import pytorch_lightning as pl
import pickle
from matplotlib import pyplot as plt

if __name__ == "__main__":
    model = DigitRecogniser()
    data = MNISTDataModule(batch_size=128)

    trainer = pl.Trainer(gpus=1)
    lr_finder = trainer.tuner.lr_find(model, data, min_lr=1e-5, max_lr=1e2)

    with open(f"lr_finder_results.pkl", "wb") as f:
        pickle.dump(lr_finder.results, f)

    plt.plot(lr_finder.results["lr"], lr_finder.results["loss"])
    plt.xscale("log")
    # plt.ylim(1, 3)
    plt.show()
    print(model)
