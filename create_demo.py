import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from main_dino import create_model
from test_ua_dino import Args
from pathlib import Path
import cv2


def enhance_attention_map(attention_map):
    """Enhance the attention map for better visualization"""
    # Convert to numpy and normalize
    att_map = attention_map.cpu().numpy()

    # Normalize to [0, 1]
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

    # Apply histogram equalization
    att_map = (att_map * 255).astype(np.uint8)
    att_map = cv2.equalizeHist(att_map)

    # Apply Gaussian blur to smooth
    att_map = cv2.GaussianBlur(att_map, (3, 3), 0)

    # Enhance contrast
    att_map = cv2.convertScaleAbs(att_map, alpha=1.5, beta=0)

    # Normalize back to [0, 1]
    att_map = att_map.astype(float) / 255

    return att_map


def create_segmentation_demo():
    # Load model
    model_dir = Path('./models')
    latest_model = max(model_dir.glob('ua_dino_*.pth'))

    args = Args()
    model = create_model(args, is_student=True)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and process image
    test_dir = Path('./test_images')
    test_image = next(test_dir.glob('*.jpg'))

    # First resize the original image
    resize_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    # Transform for model input
    model_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and resize image
    image = Image.open(test_image).convert('RGB')
    resized_image = resize_transform(image)
    img_tensor = model_transform(resized_image).unsqueeze(0)

    # Get predictions and features
    with torch.no_grad():
        mean, variance = model(img_tensor)
        features = model.backbone.get_intermediate_layers(img_tensor)[0]
        features = features[:, 1:, :]  # Remove CLS token

        # Create enhanced attention-based segmentation
        feature_map = features[0].mean(dim=1)
        attention_map = feature_map.reshape(14, 14)
        attention_map = torch.nn.functional.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear'
        )[0, 0]

        # Enhance attention map
        enhanced_attention = enhance_attention_map(attention_map)

        # Create visualization
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(resized_image)
        plt.title('Original Image', fontsize=12, pad=10)
        plt.axis('off')

        # Enhanced attention map
        plt.subplot(1, 3, 2)
        plt.imshow(enhanced_attention, cmap='magma')
        plt.title('Enhanced Attention Map', fontsize=12, pad=10)
        plt.axis('off')
        plt.colorbar(label='Attention Strength')

        # Segmentation overlay
        plt.subplot(1, 3, 3)
        img_array = np.array(resized_image)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(
            (enhanced_attention * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create overlay with better visibility
        alpha = 0.6
        overlay = (alpha * img_array + (1-alpha) * heatmap).astype(np.uint8)
        plt.imshow(overlay)
        plt.title('Attention Overlay', fontsize=12, pad=10)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('segmentation_demo.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nDemo created successfully!")
        print("Visualization shows:")
        print("1. Original image")
        print("2. Enhanced attention map (where model focuses)")
        print("3. Overlay showing potential segmentation regions")


if __name__ == "__main__":
    create_segmentation_demo()
