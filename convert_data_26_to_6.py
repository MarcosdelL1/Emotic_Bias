"""
Script to convert preprocessed data from 26 emotion categories to 6 emotion categories.
This script loads the existing .npy files and converts the categorical labels from 26 to 6 categories.
"""

import numpy as np
import os
import argparse
from emotion_mapping import convert_labels_26_to_6, create_mapping_dictionaries

def convert_data_files(data_path, output_path=None):
    """
    Convert all categorical label files from 26 to 6 categories.
    
    Args:
        data_path: Path to directory containing the original .npy files
        output_path: Path to directory where converted files will be saved (default: same as data_path)
    """
    if output_path is None:
        output_path = data_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Files to convert
    files_to_convert = [
        'train_cat_arr.npy',
        'val_cat_arr.npy', 
        'test_cat_arr.npy'
    ]
    
    print("Converting emotion categories from 26 to 6...")
    print(f"Input directory: {data_path}")
    print(f"Output directory: {output_path}")
    
    for filename in files_to_convert:
        input_file = os.path.join(data_path, filename)
        output_file = os.path.join(output_path, filename)
        
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
            
        print(f"\nConverting {filename}...")
        
        # Load the 26-category labels
        labels_26 = np.load(input_file)
        print(f"  Original shape: {labels_26.shape}")
        
        # Convert to 6 categories
        labels_6 = convert_labels_26_to_6(labels_26)
        print(f"  Converted shape: {labels_6.shape}")
        
        # Save the converted labels
        np.save(output_file, labels_6)
        print(f"  Saved to: {output_file}")
        
        # Show some statistics
        print(f"  Original 26-category stats:")
        print(f"    Non-zero labels per sample: {np.sum(labels_26 > 0, axis=1).mean():.2f} ± {np.sum(labels_26 > 0, axis=1).std():.2f}")
        print(f"  Converted 6-category stats:")
        print(f"    Non-zero labels per sample: {np.sum(labels_6 > 0, axis=1).mean():.2f} ± {np.sum(labels_6 > 0, axis=1).std():.2f}")
    
    # Copy other files that don't need conversion
    other_files = [
        'train_context_arr.npy',
        'train_body_arr.npy', 
        'train_cont_arr.npy',
        'val_context_arr.npy',
        'val_body_arr.npy',
        'val_cont_arr.npy',
        'test_context_arr.npy',
        'test_body_arr.npy',
        'test_cont_arr.npy',
        'train.csv',
        'val.csv',
        'test.csv'
    ]
    
    print(f"\nCopying other files...")
    for filename in other_files:
        input_file = os.path.join(data_path, filename)
        output_file = os.path.join(output_path, filename)
        
        if os.path.exists(input_file):
            import shutil
            shutil.copy2(input_file, output_file)
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: {filename} not found, skipping...")
    
    print(f"\nConversion complete!")
    print(f"Converted files are available in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert preprocessed data from 26 to 6 emotion categories')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to directory containing original .npy files')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to directory where converted files will be saved (default: same as data_path)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist!")
        return
    
    convert_data_files(args.data_path, args.output_path)

if __name__ == "__main__":
    main()



