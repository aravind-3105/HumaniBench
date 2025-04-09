
import os
import json
import glob
import uuid
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from ultralytics import YOLO
from datasets import load_dataset
# from IPython.display import display, HTML
from huggingface_hub import login
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import os
login(token='') # Add your Hugging Face token here if needed


# In[ ]:


#@title Vars
IMAGE_DIR = 'images' #@param
OUTPUT_JSON = 'yolo_detections.json' #@param
NUM_IMAGES = 2500 #@param
YOLO_MODEL = 'yolov8n.pt' #@param
CONFIDENCE_THRESHOLD = 0.6 #@param

os.makedirs(IMAGE_DIR, exist_ok=True)


# In[ ]:


#@title Extract images from NMB+
#@markdown I couldn't get this to preserve original filenames from hf

print("Loading the dataset...")
ds = load_dataset("vector-institute/newsmediabias-plus-clean")

random_records = ds['train'].shuffle(seed=42).select(range(NUM_IMAGES))

print(f"Extracting images from {len(random_records)} records...")
extracted_count = 0
filename_mapping = {}

for i, record in enumerate(tqdm(random_records)):
    image_data = record.get('image', None)

    if image_data is not None:
        try:
            original_filename = None
        
            if not original_filename:
                # print("No filename found")
                original_filename = f"{uuid.uuid4().hex[:10]}.jpg"

            # check filename is unique
            base_filename = original_filename
            counter = 1
            while os.path.exists(os.path.join(IMAGE_DIR, original_filename)):
                name, ext = os.path.splitext(base_filename)
                original_filename = f"{name}_{counter}{ext}"
                counter += 1

            image_path = os.path.join(IMAGE_DIR, original_filename)

            # convert image types to PIL and save
            if isinstance(image_data, dict) and 'bytes' in image_data:
                from io import BytesIO
                img = Image.open(BytesIO(image_data['bytes']))
                img.save(image_path)
                extracted_count += 1
            elif hasattr(image_data, 'convert'):
                image_data.save(image_path)
                extracted_count += 1
            elif isinstance(image_data, np.ndarray):
                img = Image.fromarray(image_data)
                img.save(image_path)
                extracted_count += 1
            else:
                print(f"Skipping record {i+1}: Unsupported image format {type(image_data)}")
                continue

            # store metadata for later
            filename_mapping[original_filename] = {
                'record_id': i,
                'title': record.get('title', ''),
                'url': record.get('image_url', '')
            }

        except Exception as e:
            print(f"Error extracting image for record {i+1}: {e}")

# save filename map for reference
with open(os.path.join(IMAGE_DIR, 'filename_mapping.json'), 'w') as f:
    json.dump(filename_mapping, f, indent=4)

print(f"Extracted {extracted_count} images to {IMAGE_DIR}")
print(f"Filename mapping saved to {os.path.join(IMAGE_DIR, 'filename_mapping.json')}")


# In[ ]:


#@title Run Yolo on Images
model = YOLO(YOLO_MODEL)

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

def process_image(image_path):
    image_id = os.path.basename(image_path)
    img_width, img_height = get_image_dimensions(image_path)

    results = model(image_path)

    annotations = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])

            # Only include detections with confidence above threshold
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                annotation = {
                    "class_name": class_name,
                    "bbox": [
                        float(x1),  # top left x coord
                        float(y1),  # top left y coord
                        float(x2),  # bottom right x coord
                        float(y2)   # bottom right y coord
                    ],
                    "confidence": conf
                }

                annotations.append(annotation)

    image_entry = {
        "id": image_id,
        "image_path": f"/{image_id}",
        "original_filename": image_id,
        "img_width": img_width,
        "img_height": img_height,
        "annotations": annotations
    }

    return image_entry



image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
print(f"Found {len(image_paths)} images to process")

results = []
for image_path in tqdm(image_paths):
    try:
        image_entry = process_image(image_path)
        results.append(image_entry)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=4)

# Get some stats about the results
total_images = len(results)
total_detections = sum(len(r['annotations']) for r in results)
images_with_detections = sum(1 for r in results if len(r['annotations']) > 0)

print(f"Processing complete. Results saved to {OUTPUT_JSON}")
print(f"Processed {total_images} images successfully")
print(f"Found {total_detections} objects with confidence ≥ {CONFIDENCE_THRESHOLD}")
print(f"{images_with_detections} images ({images_with_detections/total_images*100:.1f}%) have detections above the threshold")


# In[ ]:


#@title Visualize a few results

# Can change to 2500 for full dataset
NUM_TO_VISUALIZE = 5 #@param 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import os

with open(OUTPUT_JSON, 'r') as f:
    results = json.load(f)
print(f"Successfully loaded {len(results)} detection results from {OUTPUT_JSON}")

def visualize_detection(image_entry):
    img_path = os.path.join(IMAGE_DIR, image_entry['id'])
    try:
        img = plt.imread(img_path)
    except FileNotFoundError:
        print(f"Image file not found: {img_path}")
        return

    fig, ax = plt.subplots(1, figsize=(12, 9))

    ax.imshow(img)

    # add bounding boxes
    for ann in image_entry['annotations']:
        x1, y1, x2, y2 = ann['bbox']

        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        plt.text(x1, y1-5, f"{ann['class_name']} ({ann['confidence']:.2f})",
                 color='white', fontsize=12,
                 bbox=dict(facecolor='red', alpha=0.5))

    # display article title with original filename and any available metadata
    title = f"Detections for {image_entry['id']}"
    if 'metadata' in image_entry and 'title' in image_entry['metadata'] and image_entry['metadata']['title']:
        title += f"\nArticle: {image_entry['metadata']['title'][:50]}..."

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if len(results) > 0:
    print(f"Visualizing detections for {min(NUM_TO_VISUALIZE, len(results))} random images...")
    samples = random.sample(results, min(NUM_TO_VISUALIZE, len(results)))
    for sample in samples:
        visualize_detection(sample)
else:
    print("No results to visualize.")

# helpful info
if len(results) > 0:
    total_detections = sum(len(r['annotations']) for r in results)

    class_counts = {}
    for r in results:
        for ann in r['annotations']:
            class_name = ann['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\nDetection Statistics:")
    print(f"Total images: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(results):.2f}")

    print("\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")

