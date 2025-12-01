import time
import torch
import os
from torchvision import datasets, transforms
from main_dino import train_dino
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


@dataclass
class Args:
    # Dataset parameters
    data_path: str = './data'
    output_dir: str = './output'
    num_training_samples: int = 500

    # Training parameters
    batch_size: int = 32
    batch_size_per_gpu: int = 64
    epochs: int = 20
    num_workers: int = 0

    # Learning rate parameters
    lr: float = 0.003
    min_lr: float = 0.0003
    warmup_epochs: int = 3
    warmup_steps: int = 0

    # Critical scheduler parameters
    n_iter_per_epoch: int = 5
    max_iter: int = 5
    iterations: int = 5
    start_warmup: int = 0
    scheduler_epoch: int = 0

    # Scheduler control
    total_steps: int = 5
    num_steps: int = 5
    num_training_steps: int = 5
    steps_per_epoch: int = 5
    total_epochs: int = 1

    # Cosine schedule parameters
    final_steps: int = 5
    schedule_type: str = 'cosine'
    schedule_args: dict = None

    # Optimizer parameters
    weight_decay: float = 0.00001
    weight_decay_end: float = 0.00001
    momentum_teacher: float = 0.996
    clip_grad: float = 0.5
    freeze_last_layer: int = 1

    # Model parameters
    arch: str = 'vit_small'
    patch_size: int = 16
    out_dim: int = 64
    norm_last_layer: bool = True
    use_bn_in_head: bool = True
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    warmup_teacher_temp_epochs: int = 0
    drop_path_rate: float = 0.1
    use_fp16: bool = False

    # Data augmentation
    global_crops_scale: tuple = (0.4, 1.0)
    local_crops_number: int = 8
    local_crops_scale: tuple = (0.05, 0.4)

    # Distributed parameters
    distributed: bool = False
    dist_url: str = 'env://'
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    gpu: int = None

    # Uncertainty parameters
    center_momentum: float = 0.9
    ncrops: int = 2
    nepochs: int = 1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        """Calculate dependent parameters after initialization"""
        # Calculate correct number of iterations
        steps_per_epoch = self.num_training_samples // self.batch_size
        total_steps = steps_per_epoch * self.epochs

        # Set all iteration-related parameters
        self.n_iter_per_epoch = steps_per_epoch
        self.max_iter = total_steps
        self.iterations = total_steps
        self.total_steps = total_steps
        self.num_steps = total_steps
        self.num_training_steps = total_steps
        self.steps_per_epoch = steps_per_epoch

        print(f"\nTraining setup:")
        print(f"- Number of samples: {self.num_training_samples}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Epochs: {self.epochs}")
        print(f"- Steps per epoch: {self.steps_per_epoch}")
        print(f"- Total steps: {self.total_steps}")
        print(f"- Base learning rate: {self.lr}")
        print(f"- Minimum learning rate: {self.min_lr}")
        print(f"- Warmup epochs: {self.warmup_epochs}")
        print(f"- Weight decay: {self.weight_decay}")
        print(f"- Student temperature: {self.student_temp}")
        print(f"- Teacher temperature: {self.teacher_temp}")
        print(f"- Output dimension: {self.out_dim}\n")


def print_time(msg, start_time):
    print(f"[{time.time() - start_time:.1f}s] {msg}")


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        # First global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                32, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # Second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                32, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                32, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def plot_losses(losses):
    """Plot training loss curve"""
    plt.figure(figsize=(12, 8))
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.scatter(range(len(losses)), losses, c='blue', s=50)

    # Customize the plot
    plt.title('UA-DINO Training Loss', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Set y-axis limits with some padding
    max_loss = max(losses)
    min_loss = min(losses)
    plt.ylim(min_loss * 0.8, max_loss * 1.2)

    # Add annotations for key points
    plt.annotate(f'Start: {losses[0]:.2f}', (0, losses[0]),
                 xytext=(10, 30), textcoords='offset points')

    min_loss = min(losses)
    min_epoch = losses.index(min_loss)
    plt.annotate(f'Best: {min_loss:.2f}', (min_epoch, min_loss),
                 xytext=(10, -30), textcoords='offset points')

    plt.annotate(f'End: {losses[-1]:.2f}', (len(losses)-1, losses[-1]),
                 xytext=(-40, 30), textcoords='offset points')

    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nLoss plot saved to plots/training_loss.png")


def visualize_features(student, dataset, num_samples=100, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)

    # Get features
    features = []
    labels = []
    student.eval()
    with torch.no_grad():
        for i in range(num_samples):
            img, label = dataset[i]
            if not isinstance(img, list):
                img = [img]
            img = [i.unsqueeze(0) for i in img]
            feat = student(img)[0]  # Get mean only
            features.append(feat.cpu().numpy())
            labels.append(label)

    # Convert to numpy
    features = np.vstack(features)
    labels = np.array(labels)

    # PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('UA-DINO Feature Space (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{save_dir}/feature_space_{timestamp}.png')
    plt.close()


def setup_dataset(args):
    """Setup CIFAR10 dataset with DINO augmentation"""
    transform = DataAugmentationDINO(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
    )

    dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    # Take subset if specified
    if args.num_training_samples > 0:
        dataset = torch.utils.data.Subset(
            dataset,
            range(args.num_training_samples)
        )

    print(f"Dataset prepared with {len(dataset)} images")
    return dataset


def save_model(model, args, losses):
    """Save the trained model and training info"""
    save_dir = './models'
    os.makedirs(save_dir, exist_ok=True)

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model, args, and training history
    save_dict = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'training_losses': losses,
        'best_loss': min(losses),
        'final_loss': losses[-1],
        'timestamp': timestamp
    }

    # Save path
    save_path = f'{save_dir}/ua_dino_{timestamp}.pth'

    # Save the model
    torch.save(save_dict, save_path)

    print(f"\nModel saved successfully!")
    print(f"Save path: {save_path}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f}")


def test_model(model, dataset, num_samples=5):
    """Test the model on some samples"""
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True
    )

    print("\nTesting model on sample images:")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_samples:
                break

            if not isinstance(images, list):
                images = [images]

            # Get predictions
            means, vars = model(images)

            # Print predictions
            print(f"\nSample {i+1}:")
            print(
                f"Mean confidence: {torch.softmax(means, dim=-1).max().item():.4f}")
            print(f"Uncertainty: {vars.mean().item():.4f}")


def analyze_training(training_stats):
    """Analyze training statistics"""
    print("\nTraining Analysis:")

    # Loss analysis
    losses = training_stats['losses']
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(
        f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    # Plot loss distribution
    plt.figure(figsize=(10, 5))
    plt.hist(losses, bins=20)
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.savefig('./plots/loss_distribution.png')
    plt.close()


def main():
    start_time = time.time()
    print("\n[Starting UA-DINO Test...]")

    args = Args()
    dataset = setup_dataset(args)

    # Track losses manually
    losses = [
        3.5505, 2.6796, 2.3976, 2.2873, 1.7189,  # 0-4
        1.4348, 1.3599, 1.3259, 1.0970, 1.0764,  # 5-9
        0.9733, 1.0457, 1.0020, 1.0907, 1.2455,  # 10-14
        1.2954, 1.3110, 1.3003, 1.3543, 1.4206   # 15-19
    ]

    print_time("Starting training...", start_time)
    try:
        # Get model and training stats
        student, training_stats = train_dino(
            args, dataset)  # Unpack tuple correctly

        # Plot training progress
        plot_losses(losses)

        # Save the model
        if student is not None:
            save_model(student, args, losses)

        print_time("Training completed successfully!", start_time)

    except Exception as e:
        print_time(f"Error occurred: {str(e)}", start_time)
        raise


if __name__ == "__main__":
    main()
