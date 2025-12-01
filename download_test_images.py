import os
from pathlib import Path
from PIL import Image
import numpy as np


def create_test_images():
    """Create synthetic test images"""
    test_dir = Path('./test_images')
    test_dir.mkdir(exist_ok=True)

    # Create different test patterns
    size = (256, 256)

    # Pattern 1: Gradient
    gradient = np.linspace(
        0, 255, size[0]*size[1]).reshape(size).astype('uint8')
    Image.fromarray(gradient).save(test_dir / 'gradient.jpg')

    # Pattern 2: Checkerboard
    checkerboard = np.zeros(size, dtype='uint8')
    checkerboard[::2, ::2] = 255
    checkerboard[1::2, 1::2] = 255
    Image.fromarray(checkerboard).save(test_dir / 'checkerboard.jpg')

    # Pattern 3: Circles
    circle = np.zeros(size, dtype='uint8')
    center = (size[0]//2, size[1]//2)
    for i in range(size[0]):
        for j in range(size[1]):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            circle[i, j] = int(255 * (1 - dist/size[0]))
    Image.fromarray(circle).save(test_dir / 'circle.jpg')

    # Pattern 4: Stripes
    stripes = np.zeros(size, dtype='uint8')
    stripes[:, ::20] = 255
    Image.fromarray(stripes).save(test_dir / 'stripes.jpg')

    print(f"\nCreated 4 test images in {test_dir}")


if __name__ == "__main__":
    print("Creating test images...")
    create_test_images()
    print("Done!")
