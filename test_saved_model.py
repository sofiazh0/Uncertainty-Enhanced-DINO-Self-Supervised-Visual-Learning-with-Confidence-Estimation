import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from main_dino import create_model
from test_ua_dino import Args


def load_model(model_path):
    """Load the saved UA-DINO model"""
    print(f"\nLoading model from {model_path}")
    try:
        checkpoint = torch.load(model_path)

        # Create a new model instance
        args = Args()
        args.arch = 'vit_small'
        args.patch_size = 16
        args.out_dim = 64

        # Create model and load state dict
        model = create_model(args, is_student=True)  # Changed back to student
        model.load_state_dict(checkpoint['model_state_dict'])

        print("\nTraining setup:")
        for key, value in checkpoint['args'].items():
            print(f"- {key}: {value}")

        print("\nModel loaded successfully!")
        return model, args
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Available keys in checkpoint:", checkpoint.keys())
        raise


def prepare_image(image_path):
    """Prepare image for model input"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def get_imagenet_labels():
    """Get ImageNet class labels"""
    labels = {}
    try:
        import requests
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        if response.status_code == 200:
            labels = {i: label for i, label in enumerate(response.json())}
    except:
        print("Could not load ImageNet labels")
    return labels


def test_model():
    model_dir = Path('./models')
    model_files = list(model_dir.glob('ua_dino_*.pth'))
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {latest_model}")

    # Load model
    model, args = load_model(latest_model)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_dir = Path('./test_images')
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

    # Load ImageNet labels
    labels = get_imagenet_labels()

    for img_path in test_images:
        try:
            print(f"\nProcessing {img_path.name}")

            # Prepare image
            img = prepare_image(img_path)
            img = img.to(device)

            # Get predictions and features
            with torch.no_grad():
                mean, variance = model(img)
                features = model.backbone.get_intermediate_layers(img)[0]
                features = features[:, 1:, :]

                temperatures = [0.01, 0.005, 0.001]

                plt.figure(figsize=(20, 10))

                # Original image
                plt.subplot(2, 3, 1)
                plt.imshow(Image.open(img_path))
                plt.title('Input Image')
                plt.axis('off')

                # Raw logits
                plt.subplot(2, 3, 2)
                plt.bar(range(len(mean[0])), mean[0].cpu().numpy())
                plt.title('Raw Logits')

                # Feature visualization
                plt.subplot(2, 3, 3)
                feature_map = features[0].mean(dim=1).reshape(14, 14)
                plt.imshow(feature_map.cpu().numpy(), cmap='viridis')
                plt.title('Feature Activation Map')
                plt.colorbar()

                # Different temperature predictions with labels
                for i, temp in enumerate(temperatures):
                    scaled_mean = mean[0] / temp
                    probs = torch.softmax(scaled_mean, dim=0)

                    plt.subplot(2, 3, i + 4)

                    top_probs, top_idx = torch.topk(probs, k=5)

                    plt.bar(range(5), top_probs.cpu().numpy())
                    plt.title(f'T={temp}\nMax prob={probs.max().item():.4f}')

                    # Add class labels if available
                    x_labels = []
                    for idx in top_idx:
                        idx = idx.item()
                        if idx in labels:
                            label = labels[idx]
                            # Truncate long labels
                            if len(label) > 15:
                                label = label[:12] + "..."
                            x_labels.append(f"{label}\n({idx})")
                        else:
                            x_labels.append(f"Class {idx}")

                    plt.xticks(range(5), x_labels, rotation=45, ha='right')

                plt.tight_layout()
                plt.show()

                # Print detailed predictions
                print("\nDetailed predictions:")
                for temp in temperatures:
                    scaled_mean = mean[0] / temp
                    probs = torch.softmax(scaled_mean, dim=0)
                    top_probs, top_idx = torch.topk(probs, k=5)

                    print(f"\nTemperature {temp}:")
                    for prob, idx in zip(top_probs, top_idx):
                        idx = idx.item()
                        label = labels.get(idx, f"Class {idx}")
                        print(f"{label} ({idx}): {prob.item():.4f}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    test_model()
