# EMOTIC 6-Category Emotion Classification

This document explains how to use the new EMOTIC emotion classification system that uses 6 emotion categories instead of the original 26 categories. **All original files are preserved** - new files with `_6cat` suffix have been created for the 6-category system.

## Overview

The original EMOTIC dataset uses 26 emotion categories. This modification groups those 26 categories into 6 more general emotion categories:

### Original 26 Categories → New 6 Categories

1. **Happiness** (8 emotions):
   - Affection, Confidence, Engagement, Esteem, Excitement, Happiness, Peace, Pleasure

2. **Sadness** (5 emotions):
   - Embarrassment, Fatigue, Sadness, Suffering, Yearning

3. **Anger** (4 emotions):
   - Anger, Annoyance, Disapproval, Disquietment

4. **Fear** (4 emotions):
   - Aversion, Doubt/Confusion, Fear, Sensitivity

5. **Surprise** (2 emotions):
   - Anticipation, Surprise

6. **Disconnection** (3 emotions):
   - Disconnection, Pain, Sympathy

## New Files Created

The following new files have been created for 6-category classification (original files are preserved):

1. **`emotion_mapping.py`** - New file containing mapping functions and emotion definitions
2. **`emotic_6cat.py`** - New model architecture for 6-category classification
3. **`emotic_dataset_6cat.py`** - New dataset classes for 6-category labels
4. **`main_6cat.py`** - New main script for 6-category training and testing
5. **`train_6cat.py`** - New training script for 6-category system
6. **`test_6cat.py`** - New testing script for 6-category evaluation
7. **`inference_6cat.py`** - New inference script for 6-category predictions
8. **`convert_data_26_to_6.py`** - New script to convert existing preprocessed data

## Usage

### 1. Convert Existing Data

If you have existing preprocessed data with 26 categories, convert it to 6 categories:

```bash
python convert_data_26_to_6.py --data_path ./Data/emotic_pre --output_path ./Data/emotic_pre_6cat
```

### 2. Train the Model

Train the model with 6-category data:

```bash
python main_6cat.py --mode train --data_path ./Data/emotic_pre --experiment_path ./experiments/6cat_experiment --epochs 50 --batch_size 6
```

Note: The batch size is set to 6 (same as the number of categories) as recommended in the original code.

### 3. Test the Model

Test the trained model:

```bash
python main_6cat.py --mode test --data_path ./Data/emotic_pre --experiment_path ./experiments/6cat_experiment
```

### 4. Run Inference

Run inference on new images:

```bash
python main_6cat.py --mode inference --inference_file sample_inference_list.txt --experiment_path ./experiments/6cat_experiment
```

## Key Changes

### Model Architecture
- The final classification layer now outputs 6 categories instead of 26
- All other layers remain the same

### Data Conversion
- The conversion script groups multiple original emotions into single new categories
- If any emotion from a group is present in the original label, the new category is marked as present
- This is a many-to-one mapping, so some information is lost in the conversion

### Evaluation
- All evaluation metrics (AP, VAD errors) now work with 6 categories
- Thresholds are calculated for 6 categories instead of 26

## Benefits of 6-Category System

1. **Simpler Classification**: Easier to interpret and use in applications
2. **Better Generalization**: Fewer categories may lead to better model performance
3. **Faster Training**: Smaller output layer means faster training and inference
4. **More Balanced**: 6 categories are more balanced than 26 (some original categories had very few samples)

## Considerations

1. **Information Loss**: Converting from 26 to 6 categories loses some fine-grained emotion information
2. **Retraining Required**: You need to retrain the model with the new 6-category data
3. **Data Conversion**: Existing preprocessed data needs to be converted using the provided script

## File Structure

```
BIAS/
├── emotion_mapping.py          # Emotion mapping utilities
├── convert_data_26_to_6.py     # Data conversion script
├── emotic_6cat.py              # New model for 6 categories
├── emotic_dataset_6cat.py      # New dataset classes for 6 categories
├── main_6cat.py                # New main script for 6 categories
├── train_6cat.py               # New training script for 6 categories
├── test_6cat.py                # New testing script for 6 categories
├── inference_6cat.py           # New inference script for 6 categories
├── emotic.py                   # Original model (26 categories) - PRESERVED
├── emotic_dataset.py           # Original dataset classes - PRESERVED
├── main.py                     # Original main script - PRESERVED
├── test.py                     # Original testing functions - PRESERVED
├── train.py                    # Original training functions - PRESERVED
├── inference.py                # Original inference functions - PRESERVED
└── README_6_CATEGORIES.md      # This file
```

## Example Usage

```python
from emotion_mapping import NEW_EMOTIONS, convert_labels_26_to_6
import numpy as np

# Get the new emotion categories
print("6 Emotion Categories:", NEW_EMOTIONS)

# Convert labels from 26 to 6 categories
labels_26 = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
labels_6 = convert_labels_26_to_6(labels_26)
print("Converted labels:", labels_6)
```

This modification maintains the core functionality of the EMOTIC system while simplifying the emotion classification task to 6 basic categories.
