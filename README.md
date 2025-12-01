# Uncertainty-Enhanced DINO: Self-Supervised Visual Learning with Confidence Estimation

A PyTorch implementation of an Uncertainty-Aware extension to the DINO (Self-Distillation with No Labels) self-supervised learning framework for Vision Transformers.

## Overview

This project extends the original [DINO framework](https://github.com/facebookresearch/dino) by Facebook AI Research with uncertainty quantification capabilities. The model learns visual representations through self-supervised learning while simultaneously estimating both aleatoric and epistemic uncertainty.

### Key Features

- **Self-Supervised Learning**: Train Vision Transformers without labeled data using the DINO framework
- **Uncertainty Quantification**: Predict both mean representations and variance estimates
- **Dual Head Architecture**: Separate branches for mean and variance prediction
- **CIFAR-10 Training**: Ready-to-use training pipeline on CIFAR-10 dataset
- **Visualization Tools**: Built-in attention map visualization and feature space analysis
- **Comprehensive Metrics**: Track total, epistemic, and aleatoric uncertainty

## Quick Start

### Prerequisites

- Python 3.6+
- PyTorch 1.7.1+
- CUDA 11.0+ (for GPU training)
- torchvision 0.8.2+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sofiazh0/Uncertainty-Enhanced-DINO-Self-Supervised-Visual-Learning-with-Confidence-Estimation.git
cd Uncertainty-Enhanced-DINO-Self-Supervised-Visual-Learning-with-Confidence-Estimation
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install matplotlib numpy scikit-learn
```

3. The CIFAR-10 dataset will be automatically downloaded on first run.

## Training

### Basic Training

Run the uncertainty-aware DINO training with default parameters:

```bash
python test_ua_dino.py
```

### Advanced Training

Customize training parameters in `test_ua_dino.py` or via the `Args` dataclass:

```python
@dataclass
class Args:
    # Dataset
    data_path: str = './data'
    num_training_samples: int = 500
    
    # Training
    batch_size: int = 32
    epochs: int = 20
    lr: float = 0.003
    min_lr: float = 0.0003
    warmup_epochs: int = 3
    
    # Model
    arch: str = 'vit_small'
    patch_size: int = 16
    out_dim: int = 64
    
    # Uncertainty
    teacher_temp: float = 0.04
    student_temp: float = 0.1
```

### Full DINO Training

For standard DINO training on ImageNet:

```bash
python main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/output
```

## Architecture

### Uncertainty Head

The `UncertaintyHead` consists of two parallel branches:

- **Mean Branch**: Predicts feature representations
  - MLP layers with GELU activation
  - Bottleneck dimension: 256
  - Hidden dimension: 2048
  
- **Variance Branch**: Predicts uncertainty estimates
  - Parallel MLP architecture
  - Exponential activation ensures positive variance
  - Captures both aleatoric and epistemic uncertainty

### Loss Function

The `UncertaintyAwareDINOLoss` combines:

1. **Main DINO Loss**: Cross-entropy between student and teacher predictions
2. **Uncertainty Loss**: MSE between student and teacher variance predictions
3. **Temperature Scheduling**: Adaptive temperature scaling during training
4. **Center Update**: Momentum-based centering for stability

## Results

### Training Performance

After 20 epochs on CIFAR-10 (500 samples):
- Initial Loss: ~3.55
- Final Loss: ~1.42
- Best Loss: ~0.97
- Convergence: Stable after ~10 epochs

### Saved Outputs

Training generates several outputs in organized directories:

```
DL Project/
├── models/              # Saved model checkpoints
│   └── ua_dino_YYYYMMDD_HHMMSS.pth
├── plots/               # Training visualizations
│   ├── training_loss.png
│   ├── loss_distribution.png
│   └── feature_space_YYYYMMDD_HHMMSS.png
└── output/              # Training logs and checkpoints
    ├── checkpoint0000.pth
    └── log.txt
```

## Model Components

### Core Files

- **`main_dino.py`**: Main training loop for UA-DINO
- **`vision_transformer.py`**: Vision Transformer implementation with uncertainty support
- **`uncertainty_head.py`**: Dual-branch prediction head for mean and variance
- **`uncertainty_loss.py`**: Custom loss function for uncertainty-aware training
- **`uncertainty_metrics.py`**: Metrics tracking for different uncertainty types
- **`test_ua_dino.py`**: Standalone testing and training script
- **`utils.py`**: Utility functions for training and evaluation

### Visualization Tools

- **`visualize_attention.py`**: Self-attention map visualization
- **`video_generation.py`**: Generate attention videos from image sequences
- **`create_demo.py`**: Create visual demonstrations

### Evaluation Scripts

- **`eval_knn.py`**: k-NN classification evaluation
- **`eval_linear.py`**: Linear probing evaluation
- **`eval_video_segmentation.py`**: Video object segmentation
- **`eval_image_retrieval.py`**: Image retrieval benchmarks
- **`eval_copy_detection.py`**: Copy detection evaluation

## Visualization

### Attention Maps

Generate self-attention visualizations:

```bash
python visualize_attention.py
```

This creates attention map visualizations showing what the model focuses on in images.

### Feature Space

The training script automatically generates PCA visualizations of the learned feature space, showing how the model clusters different classes.

### Training Progress

Loss curves are automatically plotted and saved to `plots/training_loss.png` after training.

## Example Usage

### Load Pretrained Model

```python
import torch
from vision_transformer import vit_small
from uncertainty_head import UncertaintyHead

# Load model
checkpoint = torch.load('models/ua_dino_YYYYMMDD_HHMMSS.pth')
model = vit_small(patch_size=16)
model.load_state_dict(checkpoint['model_state_dict'])

# Get predictions
model.eval()
with torch.no_grad():
    mean, variance = model(images)
    uncertainty = variance.mean()
```

### Training from Scratch

```python
from test_ua_dino import Args, setup_dataset, train_dino

# Configure training
args = Args()
args.epochs = 50
args.batch_size = 64
args.lr = 0.005

# Setup and train
dataset = setup_dataset(args)
student, stats = train_dino(args, dataset)
```

## Uncertainty Metrics

The model tracks three types of uncertainty:

1. **Total Uncertainty**: Overall prediction uncertainty
2. **Epistemic Uncertainty**: Model uncertainty (reducible with more data)
3. **Aleatoric Uncertainty**: Data uncertainty (irreducible)

Access metrics during training:

```python
metrics = uncertainty_metrics.get_metrics()
print(f"Total Uncertainty: {metrics['total_uncertainty']:.4f}")
print(f"Epistemic Uncertainty: {metrics['epistemic_uncertainty']:.4f}")
print(f"Aleatoric Uncertainty: {metrics['aleatoric_uncertainty']:.4f}")
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `epochs` | 20 | Number of training epochs |
| `lr` | 0.003 | Peak learning rate |
| `min_lr` | 0.0003 | Minimum learning rate |
| `teacher_temp` | 0.04 | Teacher temperature |
| `student_temp` | 0.1 | Student temperature |
| `out_dim` | 64 | Output dimension |
| `patch_size` | 16 | Vision Transformer patch size |
| `momentum_teacher` | 0.996 | EMA momentum for teacher |

### Data Augmentation

- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Multi-crop strategy (2 global + 8 local crops)
- Global crop scale: (0.4, 1.0)
- Local crop scale: (0.05, 0.4)

## References

This project is based on the original DINO paper:

```bibtex
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `batch_size` in Args
- Reduce `num_training_samples`
- Use smaller model: `arch='vit_tiny'`

**Slow Training:**
- Increase `num_workers` for data loading
- Enable mixed precision: `use_fp16=True`
- Use GPU if available

**Loss Not Decreasing:**
- Check learning rate (try 0.001 to 0.005)
- Verify data augmentation is working
- Increase warmup epochs
- Adjust temperature parameters

## Tips

1. **Start Small**: Begin with 500-1000 samples to verify everything works
2. **Monitor Losses**: Watch both main loss and uncertainty loss components
3. **Save Often**: Models are automatically saved with timestamps
4. **Visualize**: Check attention maps to verify model is learning
5. **Tune Temperatures**: Lower temperatures = sharper predictions

## Contributing

This is a research project. Feel free to:
- Report bugs or issues
- Suggest improvements
- Extend the framework
- Share your results

## License

This project extends the original DINO implementation which is released under the Apache 2.0 license. See the [LICENSE](dino-main/LICENSE) file for details.

## Acknowledgments

- Original DINO implementation by Facebook AI Research
- Vision Transformer architecture from "An Image is Worth 16x16 Words"
- CIFAR-10 dataset by Alex Krizhevsky
- PyTorch and torchvision teams
