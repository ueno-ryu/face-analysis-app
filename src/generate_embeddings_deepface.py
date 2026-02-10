#!/usr/bin/env python3
"""
Generate face embeddings using DeepFace
"""
import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    model_name = config['recognition'].get('model_name', 'VGG-Face')
    detector_backend = config['recognition'].get('detector_backend', 'retinaface')

    print(f"Using DeepFace with model: {model_name}, detector: {detector_backend}")

    # Create embeddings directory
    embeddings_dir = Path(config['paths']['embeddings_directory'])
    embeddings_dir.mkdir(exist_ok=True)

    samples_dir = Path(config['paths']['samples_directory'])
    person_dirs = sorted(samples_dir.glob('person_*'))

    print(f"\nFound {len(person_dirs)} person directories")

    for person_dir in tqdm(person_dirs, desc="Processing persons"):
        person_id = person_dir.name
        output_path = embeddings_dir / f"{person_id}.npy"

        # Skip if embedding already exists
        if output_path.exists():
            print(f"{person_id}: Skipping (already exists)")
            continue

        embeddings = []

        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(person_dir.glob(f'*{ext}'))

        if not image_files:
            print(f"No images found in {person_id}")
            continue

        # Process each image
        valid_count = 0
        for img_path in tqdm(image_files, desc=f"  {person_id}", leave=False):
            try:
                # Extract embeddings using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                # Use first face if multiple detected
                if embedding_objs:
                    embeddings.append(np.array(embedding_objs[0]["embedding"]))
                    valid_count += 1

            except Exception as e:
                # Skip images with errors
                continue

        if embeddings:
            output_path = embeddings_dir / f"{person_id}.npy"
            np.save(output_path, np.array(embeddings))
            print(f"{person_id}: {valid_count} valid embeddings saved")
        else:
            print(f"{person_id}: No valid embeddings generated")

    print("\nEmbedding generation complete!")

if __name__ == '__main__':
    main()
