
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class CIFAR10DenseNet(nn.Module):
    def __init__(self, num_classes: int,
        in_channels : int,
        patch_size: int,
        emb_size: int,
        img_size: int,
        depth: int
        ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.depth = depth

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.img_size, kernel_size=3, padding=1), # in_channel = 3, img_size = 32
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.img_size, self.emb_size, kernel_size=3, padding=1), # emb_size = 64
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(self.emb_size * (self.patch_size // 4) * (self.patch_size // 4), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes) # Num_classes = 10
        )

    def forward(self, x):
        return self.main(x)

if __name__ == "__main__":
    _ = CIFAR10DenseNet()