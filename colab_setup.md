# Running ML-4 Projects in Google Colab

This guide explains how to run the VQA and VizDoom projects in Google Colab.

## Quick Start Guide

### 1. Clone Repository
In a Colab notebook, run:
```python
!git clone https://github.com/YOUR_USERNAME/ML-4.git
%cd ML-4
```

### 2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### 3. Create Folders in Drive
Create these folders in your Google Drive:
- `ML4_models` - For saving trained models
- `ML4_vizdoom` - For VizDoom results

### 4. Enable GPU
- Go to Runtime → Change runtime type
- Select "GPU" as Hardware accelerator

## Running VQA Training

1. Open `ITM_Classifier-baselines/train_vqa_colab.py`
2. The script is already optimized for Colab:
   - Reduced batch size
   - Uses only 10% of training data
   - Added error handling
   - Saves models to Google Drive
   - Cleans up memory automatically

To run:
```python
!python ITM_Classifier-baselines/train_vqa_colab.py
```

## Running VizDoom Training

1. Open `VizDoom-DRL-task2/train_eval_doom_colab.py`
2. The script is optimized for Colab:
   - Reduced number of environments
   - Fewer training steps
   - Uses only CNN policy (lighter)
   - Added error handling
   - Saves to Google Drive

To run:
```python
!python VizDoom-DRL-task2/train_eval_doom_colab.py
```

## Monitoring Training

Both scripts provide:
- Progress bars with live metrics
- WandB integration (optional)
- Automatic model saving
- Clear error messages

## Troubleshooting

If you encounter errors:
1. Make sure GPU runtime is enabled
2. Verify Google Drive is mounted
3. Check that data paths are correct
4. Let the scripts handle memory management

## Notes

- Training is scaled down to work on Colab's free tier
- Models automatically save checkpoints to Google Drive
- Progress can be monitored through the output cells
- Scripts handle out-of-memory errors gracefully

The code is designed to work with Colab's default environment without requiring manual package management.