import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
checkpoint_file = "checkpoint.pth"

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the model
model = timm.create_model("resnet18", pretrained=False, num_classes=10).to(device)

# Load the trained model checkpoint
model.load_state_dict(torch.load(checkpoint_file))
model.eval()

# Evaluation loop
correct = 0
total = 0
test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_loss += nn.CrossEntropyLoss(outputs, labels.to(device), reduction='sum').item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the model on the test images: {accuracy}%")
out = {'Test loss': test_loss, 'Accuracy': accuracy}

with open('./model/eval_results.json', 'w') as f:
    json.dump(eval_results, f)
