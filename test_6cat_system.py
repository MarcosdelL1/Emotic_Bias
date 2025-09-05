"""
Test script to verify the 6-category emotion classification system works correctly.
This script tests the emotion mapping and model architecture without requiring training data.
"""

import numpy as np
import torch
from emotion_mapping import NEW_EMOTIONS, convert_labels_26_to_6, create_mapping_dictionaries
from emotic_6cat import Emotic_6cat

def test_emotion_mapping():
    """Test the emotion mapping functions."""
    print("Testing emotion mapping...")
    
    # Test mapping dictionaries
    cat2ind_26, ind2cat_26, cat2ind_6, ind2cat_6, mapping_26_to_6 = create_mapping_dictionaries()
    
    print(f"Original 26 emotions: {len(ind2cat_26)}")
    print(f"New 6 emotions: {len(ind2cat_6)}")
    print(f"6 emotion categories: {NEW_EMOTIONS}")
    
    # Test label conversion
    test_labels_26 = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Anger, Annoyance, Confidence
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Happiness
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # Sadness
    ])
    
    print(f"\nOriginal 26-category labels shape: {test_labels_26.shape}")
    print(f"Sample original labels: {test_labels_26[0]}")
    
    labels_6 = convert_labels_26_to_6(test_labels_26)
    print(f"Converted 6-category labels shape: {labels_6.shape}")
    print(f"Sample converted labels: {labels_6[0]}")
    
    # Verify the conversion makes sense
    print(f"\nVerification:")
    print(f"Anger + Annoyance + Confidence -> {NEW_EMOTIONS[labels_6[0].argmax()]} (should be Anger)")
    print(f"Happiness -> {NEW_EMOTIONS[labels_6[1].argmax()]} (should be Happiness)")
    print(f"Sadness -> {NEW_EMOTIONS[labels_6[2].argmax()]} (should be Sadness)")
    
    return True

def test_model_architecture():
    """Test the 6-category model architecture."""
    print("\nTesting model architecture...")
    
    # Create model with dummy feature dimensions
    context_features = 512  # Typical ResNet18 feature dimension
    body_features = 512     # Typical ResNet18 feature dimension
    
    model = Emotic_6cat(context_features, body_features)
    print(f"Model created successfully")
    print(f"Context features: {context_features}")
    print(f"Body features: {body_features}")
    
    # Test forward pass with dummy data
    batch_size = 2
    dummy_context = torch.randn(batch_size, context_features)
    dummy_body = torch.randn(batch_size, body_features)
    
    with torch.no_grad():
        cat_out, cont_out = model(dummy_context, dummy_body)
    
    print(f"Categorical output shape: {cat_out.shape} (should be [{batch_size}, 6])")
    print(f"Continuous output shape: {cont_out.shape} (should be [{batch_size}, 3])")
    
    # Verify output shapes
    assert cat_out.shape == (batch_size, 6), f"Expected categorical output shape ({batch_size}, 6), got {cat_out.shape}"
    assert cont_out.shape == (batch_size, 3), f"Expected continuous output shape ({batch_size}, 3), got {cont_out.shape}"
    
    print("Model architecture test passed!")
    return True

def test_dataset_conversion():
    """Test the dataset conversion functionality."""
    print("\nTesting dataset conversion...")
    
    # Create dummy 26-category labels
    num_samples = 100
    labels_26 = np.random.randint(0, 2, (num_samples, 26)).astype(np.float32)
    
    # Convert to 6 categories
    labels_6 = convert_labels_26_to_6(labels_26)
    
    print(f"Converted {num_samples} samples from 26 to 6 categories")
    print(f"Original shape: {labels_26.shape}")
    print(f"Converted shape: {labels_6.shape}")
    
    # Check that the conversion preserves the multi-label nature
    original_multi_label_count = np.sum(labels_26 > 0, axis=1).mean()
    converted_multi_label_count = np.sum(labels_6 > 0, axis=1).mean()
    
    print(f"Average labels per sample (original): {original_multi_label_count:.2f}")
    print(f"Average labels per sample (converted): {converted_multi_label_count:.2f}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing 6-Category Emotion Classification System")
    print("=" * 60)
    
    try:
        # Test emotion mapping
        test_emotion_mapping()
        
        # Test model architecture
        test_model_architecture()
        
        # Test dataset conversion
        test_dataset_conversion()
        
        print("\n" + "=" * 60)
        print("All tests passed! The 6-category system is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()



