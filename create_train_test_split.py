import os
import shutil
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
base_dir = os.getcwd()
split_dir = os.path.join(base_dir, "testing_dataset_splitted")
real_faces_dir = os.path.join(split_dir, "left_split")      # Contains real faces
sketches_dir = os.path.join(split_dir, "right_split")       # Contains sketches

# Create train-test directories
dataset_dir = os.path.join(base_dir, "dataset")
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Create subdirectories
train_real_dir = os.path.join(train_dir, "real_faces")
train_sketch_dir = os.path.join(train_dir, "sketches")
test_real_dir = os.path.join(test_dir, "real_faces")
test_sketch_dir = os.path.join(test_dir, "sketches")

# Create all directories
for dir_path in [dataset_dir, train_dir, test_dir, 
                train_real_dir, train_sketch_dir, 
                test_real_dir, test_sketch_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get all image numbers
image_numbers = []
for filename in os.listdir(real_faces_dir):
    if filename.endswith('_left_split.jpg'):
        num = int(filename.split('_')[0])
        image_numbers.append(num)

# Shuffle and split
random.shuffle(image_numbers)
split_idx = int(len(image_numbers) * 0.8)  # 80-20 split
train_numbers = image_numbers[:split_idx]
test_numbers = image_numbers[split_idx:]

def copy_images(numbers, source_real, source_sketch, dest_real, dest_sketch):
    for num in numbers:
        # Copy real face
        src_real = os.path.join(source_real, f"{num}_left_split.jpg")
        dst_real = os.path.join(dest_real, f"{num}.jpg")
        
        # Copy sketch
        src_sketch = os.path.join(source_sketch, f"{num}_right_split.jpg")
        dst_sketch = os.path.join(dest_sketch, f"{num}.jpg")
        
        if os.path.exists(src_real) and os.path.exists(src_sketch):
            shutil.copy2(src_real, dst_real)
            shutil.copy2(src_sketch, dst_sketch)
        else:
            print(f"Warning: Missing files for image {num}")

print("\nCreating Training Dataset...")
copy_images(train_numbers, real_faces_dir, sketches_dir, 
           train_real_dir, train_sketch_dir)
print(f"Training set created with {len(train_numbers)} images")

print("\nCreating Testing Dataset...")
copy_images(test_numbers, real_faces_dir, sketches_dir, 
           test_real_dir, test_sketch_dir)
print(f"Testing set created with {len(test_numbers)} images")

print("\nDataset splitting complete!")
print(f"Total images processed: {len(train_numbers) + len(test_numbers)}")
