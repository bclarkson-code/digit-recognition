from dataset import MNISTDataModule
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    data = MNISTDataModule(batch_size=16)
    data.setup()
    batch = next(iter(data.train_dataloader()))

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    axes = np.concatenate(axes)

    for i, img in enumerate(batch[0]):
        img = np.array(img)[0]
        img *= 0.1307
        img += 0.3081
        axes[i].imshow(img, cmap="Greys")
        axes[i].set_xticks(())
        axes[i].set_yticks(())
    plt.show()
