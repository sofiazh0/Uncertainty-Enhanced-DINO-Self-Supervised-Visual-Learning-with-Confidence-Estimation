import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def load_and_test_model(model_path, image_path):
    # Load model
    checkpoint = torch.load(model_path)
    model = checkpoint['model_state_dict']
    args = checkpoint['args']

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    # Get predictions
    model.eval()
    with torch.no_grad():
        means, vars = model(image_tensor)

    # Print results
    confidence = torch.softmax(means, dim=-1).max().item()
    uncertainty = vars.mean().item()

    print(f"\nResults for {image_path}:")
    print(f"Confidence: {confidence:.4f}")
    print(f"Uncertainty: {uncertainty:.4f}")


if __name__ == "__main__":
    model_path = "./models/ua_dino_latest.pth"
    image_path = "./test_images/test1.jpg"
    load_and_test_model(model_path, image_path)
