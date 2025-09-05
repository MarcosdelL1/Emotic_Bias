"""
Emotion mapping utilities for reducing 26 emotion categories to 6 categories.
This module provides functions to convert between the original 26-category system
and the new 6-category system.
"""

# Original 26 emotion categories (from EMOTICS dataset)
ORIGINAL_EMOTIONS = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 
                     'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 
                     'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 
                     'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

# New 6 emotion categories (grouped from original 26)
NEW_EMOTIONS = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disconnection']

# Mapping from original 26 categories to new 6 categories
EMOTION_MAPPING = {
    # Happiness group (positive emotions)
    'Affection': 'Happiness',
    'Confidence': 'Happiness', 
    'Engagement': 'Happiness',
    'Esteem': 'Happiness',
    'Excitement': 'Happiness',
    'Happiness': 'Happiness',
    'Peace': 'Happiness',
    'Pleasure': 'Happiness',
    
    # Sadness group (negative emotions - sadness)
    'Embarrassment': 'Sadness',
    'Fatigue': 'Sadness',
    'Sadness': 'Sadness',
    'Suffering': 'Sadness',
    'Yearning': 'Sadness',
    
    # Anger group (negative emotions - anger)
    'Anger': 'Anger',
    'Annoyance': 'Anger',
    'Disapproval': 'Anger',
    'Disquietment': 'Anger',
    
    # Fear group (negative emotions - fear)
    'Aversion': 'Fear',
    'Doubt/Confusion': 'Fear',
    'Fear': 'Fear',
    'Sensitivity': 'Fear',
    
    # Surprise group (neutral/alert emotions)
    'Anticipation': 'Surprise',
    'Surprise': 'Surprise',
    
    # Disconnection group (social/relational emotions)
    'Disconnection': 'Disconnection',
    'Pain': 'Disconnection',
    'Sympathy': 'Disconnection'
}

def create_mapping_dictionaries():
    """
    Create mapping dictionaries for converting between 26 and 6 category systems.
    
    Returns:
        tuple: (cat2ind_26, ind2cat_26, cat2ind_6, ind2cat_6, mapping_26_to_6)
    """
    # Original 26 categories
    cat2ind_26 = {}
    ind2cat_26 = {}
    for idx, emotion in enumerate(ORIGINAL_EMOTIONS):
        cat2ind_26[emotion] = idx
        ind2cat_26[idx] = emotion
    
    # New 6 categories
    cat2ind_6 = {}
    ind2cat_6 = {}
    for idx, emotion in enumerate(NEW_EMOTIONS):
        cat2ind_6[emotion] = idx
        ind2cat_6[idx] = emotion
    
    # Create mapping from 26 to 6 categories
    mapping_26_to_6 = {}
    for orig_idx, orig_emotion in enumerate(ORIGINAL_EMOTIONS):
        new_emotion = EMOTION_MAPPING[orig_emotion]
        new_idx = cat2ind_6[new_emotion]
        mapping_26_to_6[orig_idx] = new_idx
    
    return cat2ind_26, ind2cat_26, cat2ind_6, ind2cat_6, mapping_26_to_6

def convert_labels_26_to_6(labels_26):
    """
    Convert labels from 26-category system to 6-category system.
    
    Args:
        labels_26: numpy array of shape (N, 26) with binary labels for 26 categories
        
    Returns:
        numpy array of shape (N, 6) with binary labels for 6 categories
    """
    _, _, _, _, mapping_26_to_6 = create_mapping_dictionaries()
    
    labels_6 = np.zeros((labels_26.shape[0], 6), dtype=labels_26.dtype)
    
    for orig_idx, new_idx in mapping_26_to_6.items():
        # If any of the original emotions in this group is present, mark the new category as present
        labels_6[:, new_idx] = np.logical_or(labels_6[:, new_idx], labels_26[:, orig_idx])
    
    return labels_6

def convert_labels_6_to_26(labels_6):
    """
    Convert labels from 6-category system back to 26-category system.
    This is a lossy conversion as we can't recover the original 26-category information.
    
    Args:
        labels_6: numpy array of shape (N, 6) with binary labels for 6 categories
        
    Returns:
        numpy array of shape (N, 26) with binary labels for 26 categories
    """
    _, _, _, _, mapping_26_to_6 = create_mapping_dictionaries()
    
    labels_26 = np.zeros((labels_6.shape[0], 26), dtype=labels_6.dtype)
    
    for orig_idx, new_idx in mapping_26_to_6.items():
        # If the new category is present, mark all original emotions in this group as present
        labels_26[:, orig_idx] = labels_6[:, new_idx]
    
    return labels_26

if __name__ == "__main__":
    import numpy as np
    
    # Test the mapping functions
    print("Original 26 emotions:")
    for i, emotion in enumerate(ORIGINAL_EMOTIONS):
        print(f"{i:2d}: {emotion}")
    
    print("\nNew 6 emotions:")
    for i, emotion in enumerate(NEW_EMOTIONS):
        print(f"{i}: {emotion}")
    
    print("\nMapping from 26 to 6 categories:")
    cat2ind_26, ind2cat_26, cat2ind_6, ind2cat_6, mapping_26_to_6 = create_mapping_dictionaries()
    for orig_idx, new_idx in mapping_26_to_6.items():
        print(f"{orig_idx:2d}: {ind2cat_26[orig_idx]:15s} -> {new_idx}: {ind2cat_6[new_idx]}")
    
    # Test conversion
    print("\nTesting label conversion:")
    test_labels_26 = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # Anger, Annoyance, Confidence
    print(f"Original 26-category labels: {test_labels_26}")
    
    labels_6 = convert_labels_26_to_6(test_labels_26)
    print(f"Converted 6-category labels: {labels_6}")
    
    labels_26_back = convert_labels_6_to_26(labels_6)
    print(f"Back-converted 26-category labels: {labels_26_back}")



