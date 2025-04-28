# Running ML-4 Projects in Google Colab

This guide provides instructions for running both the ITM Classifier and VizDoom projects in Google Colab.

## GitHub Repository Setup

When pushing to GitHub:
1. Do NOT push large datasets (visual7w-images/, visual7w-text/)
2. Do NOT push generated files (models, logs, videos)
3. Do NOT push embeddings files (*.pkl)

The .gitignore file is configured to exclude these files automatically.

### What to Push
```
ML-4/
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── ITM_Classifier-baselines/
│   ├── train_vqa.py           # Original training script
│   ├── train_vqa_colab.py     # Colab-optimized version
│   ├── train_vqa.ipynb        # Colab notebook
│   └── vqa_models.py          # Model definitions
└── VizDoom-DRL-task2/
    ├── train_eval_doom.py     # Original training script
    ├── train_eval_doom_colab.py # Colab-optimized version
    ├── train_doom.ipynb       # Colab notebook
    └── doom_agents.py         # Agent definitions
```

### Dataset Management
Instead of pushing datasets to GitHub:
1. Host the Visual7W dataset separately (e.g., Google Drive)
2. Document the data download instructions
3. Use data loading scripts that can fetch from the hosted location

## Project Structure

The repository contains two main projects:
1. ITM Classifier - Visual Question Answering (VQA) task
2. VizDoom - Deep Reinforcement Learning task

## Prerequisites

1. A Google account with access to Google Colab
2. A Weights & Biases (wandb) account for experiment tracking
3. Sufficient Google Drive storage for saving models and data

## Setup Instructions

### 1. GitHub Clone
```python
!git clone https://github.com/YOUR_USERNAME/ML-4.git
%cd ML-4
```

### 2. Google Drive Setup

1. Create the following directory structure in your Google Drive:
```
ML4_projects/
├── itm_classifier/
│   ├── models/
│   ├── logs/
│   └── data/
│       └── visual7w-images/
└── vizdoom/
    ├── models/
    ├── logs/
    └── videos/
```

2. Upload your Visual7W dataset to the `visual7w-images` directory

## Running the Projects

### A. ITM Classifier (VQA)

1. Open `ITM_Classifier-baselines/train_vqa.ipynb` in Google Colab
2. Follow the notebook instructions to:
   - Install dependencies
   - Mount Google Drive
   - Set up data paths
   - Configure wandb
   - Run training

Key features:
- GPU acceleration enabled
- Automatic model checkpointing to Google Drive
- Training progress monitoring via wandb
- Support for multiple model architectures (CNN-BERT, ViT-RoBERTa, CLIP)

### B. VizDoom

1. Open `VizDoom-DRL-task2/train_doom.ipynb` in Google Colab
2. Follow the notebook instructions to:
   - Install VizDoom and dependencies
   - Mount Google Drive
   - Configure environment
   - Set up wandb
   - Run training

Key features:
- GPU-accelerated training
- Multiple policy architectures (CNN, Transformer, Hybrid)
- Automated video recording of gameplay
- Performance metrics tracking
- Model checkpointing to Google Drive

## Important Notes

1. GPU Runtime
   - Always select "GPU" as the runtime type in Colab
   - Verify GPU availability using the provided code cells

2. Storage Management
   - Monitor Google Drive storage usage
   - Consider cleaning up old checkpoints and videos
   - Use the provided data management cells in notebooks

3. Long-Running Training
   - Consider using Colab Pro for longer sessions
   - Enable "Connect to GPU" in Colab settings
   - Use browser extensions to prevent disconnects

4. Troubleshooting
   - For VizDoom installation issues, check system dependencies
   - For CUDA errors, verify GPU runtime is enabled
   - For memory issues, reduce batch sizes
   - For wandb connection issues, ensure proper login

## Performance Considerations

1. ITM Classifier
   - Batch size optimized for Colab GPU memory
   - DataLoader workers configured for Colab environment
   - Automatic mixed precision (AMP) enabled

2. VizDoom
   - Frame skip parameter for faster training
   - Vectorized environments for parallel training
   - Efficient video recording setup
   - Memory-optimized policy architectures

## Monitoring and Visualization

Both projects include:
- Real-time training metrics
- WandB integration for experiment tracking
- Progress bars with live updates
- Automated logging and checkpointing
- Performance visualization tools