import pandas as pd
import json
import os
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Configuration
dataset_root = r".\dataset"
csv_path = os.path.join(dataset_root, "labels.csv")
target_size = 640  # Target image size
min_samples_per_class = 5  # Minimum samples per class in each split

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
valid_classes = [cls for cls in class_distribution 
                if len(class_distribution[cls]) >= 3 * min_samples_per_class]
print(f"Selected {len(valid_classes)} classes with sufficient samples")

# Second pass: process only valid classes
annotations = {}
class_files = defaultdict(list)

for filename, group in df.groupby('filename'):
    try:
        # Load original image
        original_path = os.path.join(dataset_root, 'train', filename)
        img = cv2.imread(original_path)
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
        for _, row in group.iterrows():
            try:
                region_attr = json.loads(row['region_attributes'].replace("'", '"'))
                class_name = region_attr.get('name', None)
                if class_name not in valid_classes:
                    continue
                    
                shape_attr = json.loads(row['region_shape_attributes'].replace("'", '"'))
                x, y = shape_attr['x'], shape_attr['y']
                width, height = shape_attr['width'], shape_attr['height']
                
                scaled_points = [
                    [int(x * scale), int(y * scale)],
                    [int((x + width) * scale), int(y * scale)],
                    [int((x + width) * scale), int((y + height) * scale)],
                    [int(x * scale), int((y + height) * scale)]
                ]
                
                annotations[filename]['regions'].append({
                    "transcription": class_name,
                    "points": scaled_points
                })
                class_files[class_name].append(filename)
            except:
                continue
                
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Class-aware splitting
train_files = []
val_files = []
test_files = []

for class_name in valid_classes:
    class_file_list = list(set(class_files[class_name]))  # Unique files per class
    
    # Ensure minimum samples per split
    if len(class_file_list) < 3 * min_samples_per_class:
        continue
        
    # Split files for this class
    cls_train, cls_temp = train_test_split(
        class_file_list,
        test_size=0.3,
        random_state=42
    )
    cls_val, cls_test = train_test_split(
        cls_temp,
        test_size=0.33,
        random_state=42
    )
    
    train_files.extend(cls_train)
    val_files.extend(cls_val)
    test_files.extend(cls_test)

# Remove duplicates and shuffle
train_files = list(set(train_files))
val_files = list(set(val_files))
test_files = list(set(test_files))

print(f"\nFinal split counts:")
print(f"- Train: {len(train_files)} images")
print(f"- Val: {len(val_files)} images")
print(f"- Test: {len(test_files)} images")

# Save resized images and annotations
for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    with open(os.path.join(dataset_root, f"{split_name}.txt"), 'w', encoding='utf-8') as f:
        for filename in files:
            if filename not in annotations:
                continue
                
            # Save resized image
            resized_path = os.path.join(dataset_root, 'resized', split_name, filename)
            if not os.path.exists(resized_path):
                img = cv2.imread(os.path.join(dataset_root, 'train', filename))
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
print("Train:", len(count_classes(train_files)), "classes")
print("Val:", len(count_classes(val_files)), "classes")
print("Test:", len(count_classes(test_files)), "classes")