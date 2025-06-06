import numpy as np
import pandas as pd
import json
import os
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Configuration
dataset_root = r".\dataset"
csv_path = os.path.join(dataset_root, "labels.csv")
target_size = 448  # Target image size
min_samples_per_class = 15  # Minimum samples per class in each split

# Create resized directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dataset_root, 'resized', split), exist_ok=True)

# Load and preprocess data
df = pd.read_csv(csv_path)

# First pass: collect class distribution
class_distribution = defaultdict(list)
for _, row in df.iterrows():
    try:
        region_attr = json.loads(row['region_attributes'].replace("'", '"'))
        class_name = region_attr.get('name', None)
        if class_name:
            class_distribution[class_name].append(row['filename'])
    except:
        continue

# Filter classes with insufficient samples
# valid_classes = [cls for cls in class_distribution 
#                 if len(class_distribution[cls]) >= min_samples_per_class]
valid_classes = ['Flagyl', 'Doprane', 'Toplexil']
sparse_classes = [cls for cls in class_distribution 
                if len(class_distribution[cls]) < min_samples_per_class]

print(f"Selected {len(valid_classes)} classes with sufficient samples")
print(f"Selected {len(sparse_classes)} as sparse classes")
print("=========================================")

class_counts = {}
for cls in class_distribution:
    class_counts[cls] = len(class_distribution.get(cls, 0))
max_class_occurances = np.max(list(class_counts.values()))
valid_classes.append("Unknown")

# Second pass: process valid classes
annotations = {}
class_files = {'train': defaultdict(list),
               'val': defaultdict(list),
               'test': defaultdict(list)
            }

# Load original images
for split in ["train", "test", "val"]:
    try:
        split_path = os.path.join(dataset_root, split)

        for filename in os.listdir(split_path):
            
            file_path = os.path.join(split_path, filename)
            img = cv2.imread(file_path)
            if img is None:
                continue

            # Calculate scaling
            h, w = img.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Store image info
            annotations[filename] = {
                'scaled_dim': (new_w, new_h),
                'regions': [],
                'scale_factor': scale
            }
            
            # Process regions
            filename_group = df.groupby('filename').get_group(filename)
            for _, row in filename_group.iterrows():
                try:
                    region_attr = json.loads(row['region_attributes'].replace("'", '"'))
                    class_name = region_attr.get('name', None)
                                        
                    shape_attr = json.loads(row['region_shape_attributes'].replace("'", '"'))
                    x, y = shape_attr['x'], shape_attr['y']
                    width, height = shape_attr['width'], shape_attr['height']
                    
                    scaled_points = [
                        [int(x * scale), int(y * scale)],
                        [int((x + width) * scale), int(y * scale)],
                        [int((x + width) * scale), int((y + height) * scale)],
                        [int(x * scale), int((y + height) * scale)]
                    ]
                    
                    if class_name in sparse_classes \
                            and len(class_files[split]["Unknown"]) < max_class_occurances:
                        class_files[split]["Unknown"].append(filename)
                        annotations[filename]['regions'].append({
                            "transcription": "Unknown",
                            "points": scaled_points
                        })
                    elif class_name in valid_classes:                
                        annotations[filename]['regions'].append({
                                "transcription": class_name,
                                "points": scaled_points
                            })                                        
                        class_files[split][class_name].append(filename)
                except:
                    continue
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Class-aware splitting
split_files = {'train': [], 'val': [], 'test': []}
train_classes = []

# generate train files
for class_name in valid_classes:
    class_file_list = list(set(class_files["train"][class_name]))  # Unique files per class
    
    # Ensure minimum samples per split
    if len(class_file_list) < min_samples_per_class:
        continue
    
    train_classes.append(class_name)
    split_files['train'].extend(class_file_list)
    
# generate val & test files
for split in ['val', 'test']:
    for class_name in valid_classes:
        class_file_list = list(set(class_files[split][class_name]))  # Unique files per class
        
        # Ensure class exists in train split
        if class_name not in train_classes:
            continue
        
        split_files[split].extend(class_file_list)

# Remove duplicates and shuffle
for split in ["train", "test", "val"]:
    split_files[split] = list(set(split_files[split])) 

print(f"\nFinal split counts:")
print(f"- Train: {len(split_files['train'])} images")
print(f"- Val: {len(split_files['val'])} images")
print(f"- Test: {len(split_files['test'])} images")

# Save resized images and annotations
for split_name, files in [('train', split_files['train']), ('val', split_files['val']), ('test', split_files['test'])]:
    with open(os.path.join(dataset_root, f"{split_name}.txt"), 'w', encoding='utf-8') as f:
        for filename in files:
            # if filename not in annotations:
            #     continue
                
            # Save resized image
            resized_path = os.path.join(dataset_root, 'resized', split_name, filename)
            if not os.path.exists(resized_path):
                img = cv2.imread(os.path.join(dataset_root, split_name, filename))
                if img is not None:
                    cv2.imwrite(resized_path, cv2.resize(img, 
                        annotations[filename]['scaled_dim']))
            
            # Write annotation
            img_path = os.path.join('resized', split_name, filename).replace('\\', '/')
            line = f"./{img_path}\t{json.dumps(annotations[filename]['regions'], ensure_ascii=False)}"
            f.write(line + '\n')

# Verify class distribution in splits
def count_classes(split_files):
    class_counts = defaultdict(int)
    for filename in split_files:
        if filename in annotations:
            for region in annotations[filename]['regions']:
                class_counts[region['transcription']] += 1
    return class_counts

print("\nClass distribution verification:")
print("Train:", len(count_classes(split_files['train'])), "classes")
print("Val:", len(count_classes(split_files['val'])), "classes")
print("Test:", len(count_classes(split_files['test'])), "classes")