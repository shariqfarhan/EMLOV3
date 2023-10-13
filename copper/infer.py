from typing import Tuple, List
import torch
from torchvision import transforms
from PIL import Image

import lightning as L
import torch
import hydra
from omegaconf import DictConfig

from copper import utils

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[List[str], List[float]]:
    # Load the pre-trained model (replace with your actual model)
    ckpt_path = cfg.get("ckpt_path")
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    # model = torch.load(ckpt_path)
    # model.load_state_dict(ckpt_path['model_state_dict'])
    model.eval()

    # Define a transform to preprocess the image for the model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the input image
    image_path = cfg.get("img_path")
    img = Image.open(image_path)
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img)

    # Get the top 2 class probabilities and their corresponding labels
    _, indices = torch.topk(outputs, 2)
    probs = torch.softmax(outputs, dim=1)[0]  # Softmax to get probabilities
    probs = probs[indices]  # Get the top 2 probabilities
    probs = probs.tolist()

    # Load the class labels (replace with your actual labels)
    class_labels = ["Cat", "Dog"]

    top_labels = [class_labels[i] for i in range(len(probs[0]))]
    top_probs = probs[0]

    return top_labels, top_probs

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    image_path = cfg.get("img_path")
    top_labels, top_probs = infer(cfg)
    for label, prob in zip(top_labels, top_probs):
        print(f"Class: {label}, Probability: {prob:.4f}")


if __name__ == "__main__":
    main()
