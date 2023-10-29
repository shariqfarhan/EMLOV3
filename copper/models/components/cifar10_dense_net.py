
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models

# Define the ViT model with variable patch size
class CIFAR10DenseNet(nn.Module):
    def __init__(self, num_classes: int,
        in_channels : int,
        patch_size: int,
        emb_size: int,
        img_size: int,
        depth: int
        ):
        super(CIFAR10DenseNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.depth = depth

        # Define the model architecture with variable patch size
        # For simplicity, we'll use a pre-trained ResNet-18 model as the backbone
        # You can replace this with a custom ViT architecture

        # Use ResNet-18 as a backbone
        self.backbone = models.resnet18(pretrained=True)
        # Modify the input layer to adapt to the patch size
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size, padding=1, bias=False)
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # Add a custom classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

if __name__ == "__main__":
    _ = CIFAR10DenseNet()