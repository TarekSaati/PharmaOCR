import pandas as pd
import json
import os
import cv2
from sklearn.model_selection import train_test_split

# Configuration
dataset_root = r".\dataset"
csv_path = os.path.join(dataset_root, "labels.csv")
target_size = 640  # Adjust based on your GPU capacity (try 1024, 800, or 640)

# Create resized directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dataset_root, 'resized', split), exist_ok=True)

# Load and preprocess data
df = pd.read_csv(csv_path)

# Group annotations and process images
annotations = {}
for filename, group in df.groupby('filename'):
    try:
        # Load original image
        original_path = os.path.join(dataset_root, 'train', filename)
        img = cv2.imread(original_path)
        if img is None:
            print(f"Image {filename} not found. Skipping.")
            continue
            
        # Calculate scaling factor
        h, w = img.shape[:2]
        scale = target_size / max(h, w)  # Maintain aspect ratio
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Store resized image path
        annotations[filename] = {
            'scaled_dim': (new_w, new_h),
            'regions': [],
            'scale_factor': scale
        }
        
        # Process regions
        for _, row in group.iterrows():
            # Original coordinates
            shape_attr = json.loads(row['region_shape_attributes'].replace("'", '"'))
            x = shape_attr['x']
            y = shape_attr['y']
            width = shape_attr['width']
            height = shape_attr['height']
            
            # Scale coordinates
            scaled_points = [
                [int(x * scale), int(y * scale)],
                [int((x + width) * scale), int(y * scale)],
                [int((x + width) * scale), int((y + height) * scale)],
                [int(x * scale), int((y + height) * scale)]
            ]
            
            # Get transcription
            region_attr = json.loads(row['region_attributes'].replace("'", '"'))
            transcription = region_attr.get('name', '###')
            
            annotations[filename]['regions'].append({
                "transcription": transcription,
                "points": scaled_points
            })
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Split the data
all_files = list(annotations.keys())
train_files, temp = train_test_split(
    all_files,
    test_size=0.2,  # 80% train, 20% temp (val+test)
    random_state=42
)
val_files, test_files = train_test_split(
    temp,
    test_size=0.25,  # 15% val, 5% test of total
    random_state=42
)

# Save resized images and generate annotations
for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    with open(os.path.join(dataset_root, f"{split_name}.txt"), 'w', encoding='utf-8') as f:
        for filename in files:
            # Save resized image to appropriate split directory
            resized_path = os.path.join(dataset_root, 'resized', split_name, filename)
            if not os.path.exists(resized_path):
                scaled_img = cv2.resize(
                    cv2.imread(os.path.join(dataset_root, 'train', filename)),
                    (annotations[filename]['scaled_dim'][0], 
                    annotations[filename]['scaled_dim'][1]))
                cv2.imwrite(resized_path, scaled_img)
            
            # Write annotation entry
            img_entry_path = os.path.join(dataset_root, 'resized', split_name, filename).replace('\\', '/')
            line = f"{img_entry_path}\t{json.dumps(annotations[filename]['regions'], ensure_ascii=False)}"
            f.write(line + '\n')

print(f"""\nResized dataset created:
- Target size: {target_size}px (original: 2448px)
- Memory reduction: {(target_size/2448)**2 * 100:.1f}% of original size
- Final splits:
  • Train: {len(train_files)} images
  • Val: {len(val_files)} images
  • Test: {len(test_files)} images
""")