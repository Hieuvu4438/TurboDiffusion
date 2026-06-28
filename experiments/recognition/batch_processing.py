"""
Batch Face Recognition Processing

Process multiple image pairs for face recognition comparison.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

from face_recognition import FaceRecognizer


def load_image_pairs_from_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    Load image pairs from a CSV file.
    
    CSV format:
    original_path,reference_path
    /path/to/img1.jpg,/path/to/ref1.jpg
    ...
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of (original_path, reference_path) tuples
    """
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['original_path'], row['reference_path']))
    return pairs


def load_image_pairs_from_folders(original_folder: str, 
                                   reference_folder: str) -> List[Tuple[str, str]]:
    """
    Load image pairs from two folders with matching filenames.
    
    Args:
        original_folder: Path to folder containing original images
        reference_folder: Path to folder containing reference images
        
    Returns:
        List of (original_path, reference_path) tuples
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    original_folder = Path(original_folder)
    reference_folder = Path(reference_folder)
    
    pairs = []
    
    for orig_file in sorted(original_folder.iterdir()):
        if orig_file.suffix.lower() not in valid_extensions:
            continue
        
        # Try to find matching reference file
        for ext in valid_extensions:
            ref_file = reference_folder / f"{orig_file.stem}{ext}"
            if ref_file.exists():
                pairs.append((str(orig_file), str(ref_file)))
                break
    
    return pairs


def batch_compare(recognizer: FaceRecognizer,
                  image_pairs: List[Tuple[str, str]],
                  output_dir: str,
                  save_visualizations: bool = True) -> List[dict]:
    """
    Compare multiple image pairs in batch.
    
    Args:
        recognizer: FaceRecognizer instance
        image_pairs: List of (original_path, reference_path) tuples
        output_dir: Directory to save results
        save_visualizations: Whether to save visualization images
        
    Returns:
        List of result dictionaries
    """
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(image_pairs)} image pairs...")
    print("="*60)
    
    for i, (orig_path, ref_path) in enumerate(image_pairs, 1):
        print(f"\n[{i}/{len(image_pairs)}] Processing pair:")
        print(f"  Original: {orig_path}")
        print(f"  Reference: {ref_path}")
        
        try:
            # Generate output filename
            orig_name = Path(orig_path).stem
            ref_name = Path(ref_path).stem
            vis_filename = f"{orig_name}_vs_{ref_name}.png"
            vis_path = str(vis_dir / vis_filename) if save_visualizations else None
            
            # Compare faces
            similarity, is_same = recognizer.compare_faces(
                orig_path, ref_path,
                output_path=vis_path,
                show_plot=False
            )
            
            results.append({
                'original': orig_path,
                'reference': ref_path,
                'similarity': similarity,
                'is_match': is_same,
                'threshold': recognizer.threshold,
                'visualization': vis_path
            })
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'original': orig_path,
                'reference': ref_path,
                'similarity': None,
                'is_match': None,
                'threshold': recognizer.threshold,
                'error': str(e)
            })
    
    return results


def save_results_to_csv(results: List[dict], output_path: str):
    """
    Save batch results to a CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['original', 'reference', 'similarity', 'is_match', 'threshold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'original': result['original'],
                'reference': result['reference'],
                'similarity': result.get('similarity', 'N/A'),
                'is_match': result.get('is_match', 'N/A'),
                'threshold': result['threshold']
            })
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[dict]):
    """
    Print a summary of batch results.
    """
    valid_results = [r for r in results if r.get('similarity') is not None]
    
    if not valid_results:
        print("\nNo valid results to summarize.")
        return
    
    similarities = [r['similarity'] for r in valid_results]
    matches = sum(1 for r in valid_results if r['is_match'])
    
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    print(f"Total pairs processed: {len(results)}")
    print(f"Successful comparisons: {len(valid_results)}")
    print(f"Failed comparisons: {len(results) - len(valid_results)}")
    print(f"\nSimilarity Statistics:")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std:  {np.std(similarities):.4f}")
    print(f"  Min:  {np.min(similarities):.4f}")
    print(f"  Max:  {np.max(similarities):.4f}")
    print(f"\nMatch Results:")
    print(f"  Matches: {matches} ({100*matches/len(valid_results):.1f}%)")
    print(f"  Non-matches: {len(valid_results)-matches} ({100*(len(valid_results)-matches)/len(valid_results):.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Batch Face Recognition Processing')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help='CSV file with image pairs')
    group.add_argument('--folders', nargs=2, metavar=('ORIGINAL', 'REFERENCE'),
                       help='Original and reference image folders')
    
    parser.add_argument('--output', '-o', type=str, default='./output/batch_results',
                        help='Output directory for results')
    parser.add_argument('--threshold', '-t', type=float, default=0.75,
                        help='Cosine similarity threshold (default: 0.75)')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--no-vis', action='store_true',
                        help='Do not save visualization images')
    
    args = parser.parse_args()
    
    # Load image pairs
    if args.csv:
        image_pairs = load_image_pairs_from_csv(args.csv)
    else:
        image_pairs = load_image_pairs_from_folders(args.folders[0], args.folders[1])
    
    if not image_pairs:
        print("Error: No image pairs found.")
        return
    
    print(f"Found {len(image_pairs)} image pairs to process.")
    
    # Initialize recognizer
    recognizer = FaceRecognizer(threshold=args.threshold, device=args.device)
    
    # Process batch
    results = batch_compare(
        recognizer,
        image_pairs,
        args.output,
        save_visualizations=not args.no_vis
    )
    
    # Save results
    csv_output = os.path.join(args.output, 'results.csv')
    save_results_to_csv(results, csv_output)
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
