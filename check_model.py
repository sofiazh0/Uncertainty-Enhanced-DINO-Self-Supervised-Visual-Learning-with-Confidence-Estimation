import torch
from pathlib import Path


def check_saved_model():
    # Find latest model
    model_dir = Path('./models')
    model_files = list(model_dir.glob('ua_dino_*.pth'))
    if not model_files:
        raise ValueError("No saved models found!")

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"\nChecking model: {latest_model}")

    # Load checkpoint
    checkpoint = torch.load(latest_model)

    # Print checkpoint contents
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"\n{key}:")
            for subkey in checkpoint[key].keys():
                print(f"  - {subkey}")
        else:
            print(f"\n{key}:")
            print(f"  Type: {type(checkpoint[key])}")
            if isinstance(checkpoint[key], torch.Tensor):
                print(f"  Shape: {checkpoint[key].shape}")

    # Check model state dict
    if 'model_state_dict' in checkpoint:
        print("\nModel state dict keys:")
        for key in checkpoint['model_state_dict'].keys():
            if isinstance(checkpoint['model_state_dict'][key], torch.Tensor):
                print(f"- {key}: {checkpoint['model_state_dict'][key].shape}")

    # Print training info
    if 'training_losses' in checkpoint:
        losses = checkpoint['training_losses']
        print(f"\nTraining losses:")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Best loss: {min(losses):.4f}")

    if 'args' in checkpoint:
        print("\nTraining arguments:")
        for key, value in checkpoint['args'].items():
            print(f"- {key}: {value}")


if __name__ == "__main__":
    check_saved_model()
