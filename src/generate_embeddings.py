#!/usr/bin/env python3
"""
Generate face embeddings from sample images
"""
import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from insightface.app import FaceAnalysis
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Initialize InsightFace for 0.2.1
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=tuple(config['recognition'].get('det_size', [640, 640])))

    # Create embeddings directory
    embeddings_dir = Path(config['paths']['embeddings_directory'])
    embeddings_dir.mkdir(exist_ok=True)

    samples_dir = Path(config['paths']['samples_directory'])

    # Process each person directory
    person_dirs = sorted(samples_dir.glob('person_*'))

    print(f"\n📸 Found {len(person_dirs)} person directories")

    for person_dir in tqdm(person_dirs, desc="Processing persons"):
        person_id = person_dir.name
        embeddings = []

        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(person_dir.glob(f'*{ext}'))

        if not image_files:
            print(f"⚠️  No images found in {person_id}")
            continue

        # Process each image
        valid_count = 0
        for img_path in tqdm(image_files, desc=f"  {person_id}", leave=False):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                faces = app.get(img)

                # Only use images with exactly 1 face
                if len(faces) == 1:
                    embeddings.append(faces[0].embedding)
                    valid_count += 1
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

        if embeddings:
            # Save as numpy array
            output_path = embeddings_dir / f"{person_id}.npy"
            np.save(output_path, np.array(embeddings))
            print(f"✅ {person_id}: {valid_count} valid embeddings saved")
        else:
            print(f"⚠️  {person_id}: No valid embeddings generated")

    print("\n✅ Embedding generation complete!")

if __name__ == '__main__':
    main()
