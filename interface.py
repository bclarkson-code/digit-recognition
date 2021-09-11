import pygame
from matplotlib import pyplot as plt
from PIL import Image
from model import DigitRecogniser
from glob import glob
import torch
import numpy as np
from torchvision import transforms
from torchvision.io import read_image


def process_image(path):
    img = read_image(path)
    img = 1 - (img[:1, :, :] / 255.0)
    img_transforms = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    img = img_transforms(img)
    img = torch.unsqueeze(img, 0)
    return img


def get_prediction(path, model, device):
    img = process_image(path)
    img = img.to(device)
    pred = model(img)
    pred = torch.softmax(pred, 1).detach().cpu().numpy()
    plt.bar(list(range(10)), pred[0])
    plt.xticks(list(range(10)))
    plt.show()


def get_latest_checkpoint():
    checkpoints = glob("lightning_logs/*/checkpoints/*")
    return max(checkpoints)


if __name__ == "__main__":
    # Setup window
    pygame.init()
    screen = pygame.display.set_mode((560, 560))
    pygame.display.set_caption("Digit Recogniser")
    clock = pygame.time.Clock()

    # Setup model
    checkpoint = get_latest_checkpoint()
    device = torch.device("cuda")
    model = DigitRecogniser.load_from_checkpoint(checkpoint)
    model = model.eval()
    model = model.to(device)

    # Main loop
    loop = True
    press = False
    screen.fill((255, 255, 255))
    while loop:
        try:
            # Allow quitting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    loop = False

            # Read the screen, make a prediction and erase the screen
            if event.type == pygame.KEYDOWN:
                pygame.image.save(screen, "digit.jpeg")
                get_prediction("digit.jpeg", model, device)
                screen.fill((255, 255, 255))

            # Handle drawing on the screen
            px, py = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed() == (1, 0, 0):
                pygame.draw.rect(screen, (0, 0, 0), (px, py, 75, 75))
            if event.type == pygame.MOUSEBUTTONUP:
                press == False
            pygame.display.update()
            clock.tick(1000)
        except Exception as e:
            break
