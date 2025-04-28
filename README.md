# Advanced Machine Learning - Visual QA and Game Policy Learning

This repository contains implementations for two machine learning tasks:
1. Multi-Choice Visual Question Answering
2. Deep Reinforcement Learning for ViZDoom

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.28.0
timm>=0.6.12
wandb>=0.15.0
stable-baselines3>=2.0.0
vizdoom>=1.2.0
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Task 1: Visual Question Answering

Implementation of three different architectures for matching images with question-answer pairs:

1. **CNN + BERT**: 
   - ResNet50 for visual features
   - BERT for text embeddings
   - Multi-layer fusion network

2. **ViT + RoBERTa**:
   - Vision Transformer for image processing
   - RoBERTa for text understanding
   - Cross-attention fusion mechanism

3. **CLIP Zero-Shot**:
   - OpenAI's CLIP model
   - Zero-shot classification capability

### Data Structure
```
ITM_Classifier-baselines/
├── visual7w-images/      # Image files
├── visual7w-text/        # Text data files
│   ├── v7w.TrainImages.itm.txt
│   ├── v7w.DevImages.itm.txt
│   └── v7w.TestImages.itm.txt
└── v7w.sentence_embeddings-gtr-t5-large.pkl  # Pre-computed embeddings
```

### Usage

```bash
cd ITM_Classifier-baselines

# Train and evaluate all models
python train_vqa.py

# The script will:
# 1. Load and preprocess the Visual7W dataset
# 2. Train each architecture
# 3. Track metrics using wandb
# 4. Save models and evaluate performance
```

### Metrics
- Classification Accuracy
- Mean Reciprocal Rank
- Training & Test Times

## Task 2: Game Policy Learning

Implementation of three different Deep RL agents for playing ViZDoom:

1. **CNN Policy**:
   - Convolutional layers for visual processing
   - Fully connected policy head
   - Designed for efficient frame processing

2. **Transformer Policy**:
   - Vision Transformer architecture
   - Self-attention for temporal dependencies
   - Patch-based image encoding

3. **Hybrid CNN-Transformer Policy**:
   - CNN for initial feature extraction
   - Transformer for temporal reasoning
   - Combined representation learning

### Configuration

```bash
cd VizDoom-DRL-task2

# Train all agents (with 3 random seeds each)
python train_eval_doom.py

# The script will:
# 1. Train each policy architecture with multiple seeds
# 2. Track training progress using wandb
# 3. Save trained models and metrics
# 4. Generate performance visualizations
```

### Training Parameters
```python
# Default configuration
n_envs = 8               # Number of parallel environments
total_timesteps = 1e6    # Total training steps
frame_skip = 4           # Number of frames to skip
n_seeds = 3              # Number of random seeds
eval_episodes = 20       # Number of evaluation episodes

# PPO hyperparameters
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
clip_range = 0.2
```

### Metrics
- Average Reward
- Average Game Score
- Average Steps Per Episode
- Training & Test Times
- Performance comparisons across seeds

## Results

Results are saved in:
- `ITM_Classifier-baselines/results/` for VQA models
- `VizDoom-DRL-task2/logs/` for ViZDoom agents

Each directory contains:
- Model checkpoints
- Training statistics
- Evaluation metrics
- Performance visualizations

## Visualizations

For the ViZDoom task, trained agents can be visualized:
- Training progress curves
- Reward distributions
- Episode length statistics
- Video recordings of gameplay

## WandB Integration

Both tasks use Weights & Biases for experiment tracking:
1. Create an account at wandb.ai
2. Login using: `wandb login`
3. View results at: wandb.ai/your-username/project-name

## Notes

- GPU recommended for faster training
- Models are saved at regular checkpoints
- Use multiple seeds for reliable results
- See individual task directories for detailed configurations