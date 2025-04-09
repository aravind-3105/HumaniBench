
import re
from PIL import Image
import torch
import json
import os
import time
from argparse import ArgumentParser
import logging
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import base64

# ------------------------------------------------------------------------------
# CONFIG
# Replace this with the correct Phi 4 model path:
MODEL_NAME_OR_PATH = "CohereForAI/aya-vision-8b"  # Example: LLaVA Phi 4 model

LABELS_JSON = "./results/eval4/yolo_detections.json"  # Your ground-truth label file
IMAGES_DIR = "./results/eval4/images"  # Directory with image files
OUTPUT_JSON = "llava_aya_bboxes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = ""

# How many records to process? None => all, or set a small integer (e.g., 10) for testing.
NUM_RECORDS = None  # or 10

# ------------------------------------------------------------------------------
# 1. Load ground-truth records
with open(LABELS_JSON, "r") as f:
    gt_records = json.load(f)

if NUM_RECORDS is not None:
    gt_records = gt_records[:NUM_RECORDS]

print(f"Loaded {len(gt_records)} records from {LABELS_JSON}")

# ------------------------------------------------------------------------------
# 2. Load the LLaVA Phi 4 model + processor
print(f"Loading model from {MODEL_NAME_OR_PATH}...")




model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME_OR_PATH, 
                                                        device_map="auto",
                                                        trust_remote_code=True, 
                                                        # torch_dtype=torch.float16,
                                                        cache_dir=CACHE_DIR)

processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, 
                                              trust_remote_code=True, 
                                              cache_dir=CACHE_DIR)


model.eval()


# ------------------------------------------------------------------------------
# 3. Utility functions

def stage1_prompt(class_names):
    """
    Stage 1: Ask LLaVA about the classes in free-form bounding boxes.
    """
    system_text = "You are a vision-language assistant."
    user_text = (
        "<image>\n"
        "You are given an image with one or more objects.\n"
        f"Please find the following classes:\n- {', '.join(class_names)}\n\n"
        "Provide approximate bounding boxes (in any form: normalized or pixel). "
        "Explain what you see."
    )

    prompt = f"System: {system_text}\nUser: {user_text}\nAssistant:"
    return prompt


   

def stage2_refine_prompt(stage1_text, img_width, img_height):
    """
    Stage 2: ask LLaVA to produce strict "Object: <class>, BBox: [x1, y1, x2, y2]" lines.
    """
    system_text = "You are a helpful assistant."
    user_text = (
        f"The previous assistant said:\n{stage1_text}\n\n"
        "Now please restate ONLY the bounding boxes in the exact format:\n"
        "Object: <class_name>, BBox: [x1, y1, x2, y2]\n\n"
        f"- x1, y1, x2, y2 are integer or float coordinates.\n"
        f"- If you provided normalized coords before, multiply them by the image width "
        f"({img_width}) and height ({img_height}), then round.\n"
        "No extra text."
    )

    prompt = f"System: {system_text}\nUser: {user_text}\nAssistant:"
    return prompt

def run_llava_inference(prompt_text, image=None):
    """
    """
    if image is not None:
        inputs = processor(
            text=[prompt_text],
            images=[image],  # Pass a list of length 1
            return_tensors="pt"
        ).to(DEVICE)


    else:
    # Text-only
        inputs = processor(
        text=[prompt_text],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_floats_bboxes(text):
    """
    Regex to capture lines like:
      Object: tie, BBox: [0.009, 0.373, 0.629, 0.986]
      Object: person, BBox: [100, 40, 120, 80]
    We allow float or integer (i.e., \d*\.?\d+).
    Returns a list of {class_name, bbox} in float form.
    """
    pattern = r"Object:\s*(.*?),\s*BBox:\s*\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]"
    matches = re.findall(pattern, text)
    results = []
    for (obj_name, x1_str, y1_str, x2_str, y2_str) in matches:
        x1, y1, x2, y2 = map(float, [x1_str, y1_str, x2_str, y2_str])
        results.append({
            "class_name": obj_name.strip(),
            "bbox": [x1, y1, x2, y2]
        })
    return results

def convert_to_pixel(bbox, img_width, img_height):
    """
    Convert normalized coords in [0..1] to pixel coords if they look normalized.
    If any coordinate > 2 or so, assume they're already pixel coords.
    """
    x1, y1, x2, y2 = bbox
    # naive check: if all are <= 1, assume normalized
    if (0 <= x1 <= 1.0) and (0 <= y1 <= 1.0) and (0 <= x2 <= 1.0) and (0 <= y2 <= 1.0):
        x1 = round(x1 * img_width)
        y1 = round(y1 * img_height)
        x2 = round(x2 * img_width)
        y2 = round(y2 * img_height)
    else:
        # assume they're already pixel coords
        x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
    
    # Ensure x1 < x2, y1 < y2 if that's required
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [int(x1), int(y1), int(x2), int(y2)]

def iou(boxA, boxB):
    """
    Intersection over Union for [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


# ------------------------------------------------------------------------------
# 4. Main Loop
predictions = []

for idx, record in enumerate(gt_records, start=1):
    filename = record["original_filename"]
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        print(f"[{idx}] Image not found: {image_path}, skipping.")
        continue

    annos = record.get("annotations", [])
    class_names = list({a["class_name"] for a in annos})
    if not class_names:
        print(f"[{idx}] No classes for {filename}, skipping.")
        continue

    img_width = record.get("img_width", 224)
    img_height = record.get("img_height", 224)

    print(f"[{idx}] Processing {filename} (classes={class_names})...")

    # Stage 1
    image_pil = Image.open(image_path).convert("RGB")
    prompt1 = stage1_prompt(class_names)
    stage1_out = run_llava_inference(prompt1, image=image_pil)

    # Stage 2
    prompt2 = stage2_refine_prompt(stage1_out, img_width, img_height)
    stage2_out = run_llava_inference(prompt2, image=None)

    # Parse boxes
    parsed_boxes = parse_floats_bboxes(stage2_out)
    
    # Convert to pixel coords
    llava_boxes = []
    for p in parsed_boxes:
        px_box = convert_to_pixel(p["bbox"], img_width, img_height)
        llava_boxes.append({
            "class_name": p["class_name"],
            "bbox": px_box
        })
    
    # Optionally compute best IoU for each predicted box vs. ground-truth
    for box_info in llava_boxes:
        best_iou_val = 0.0
        for gt in annos:
            if gt["class_name"] == box_info["class_name"]:
                gt_box = [
                    int(round(gt["bbox"][0])),
                    int(round(gt["bbox"][1])),
                    int(round(gt["bbox"][2])),
                    int(round(gt["bbox"][3]))
                ]
                iou_val = iou(box_info["bbox"], gt_box)
                if iou_val > best_iou_val:
                    best_iou_val = iou_val
        box_info["best_iou"] = best_iou_val

    # Build final record
    out_record = {
        "id": filename,
        "image_path": image_path,
        "original_filename": filename,
        "img_width": img_width,
        "img_height": img_height,

        "ground_truth_annotations": annos,

        "raw_stage1_output": stage1_out[:3000],
        "raw_stage2_output": stage2_out[:3000],

        "annotations": []
    }

    # Add predicted boxes
    for box_info in llava_boxes:
        out_record["annotations"].append({
            "class_name": box_info["class_name"],
            "bbox": box_info["bbox"],
            "confidence": 1.0,
            "best_iou_with_gt": round(box_info["best_iou"], 3)
        })

    predictions.append(out_record)

# ------------------------------------------------------------------------------
# 5. Save final predictions
with open(OUTPUT_JSON, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nDone! Processed {len(predictions)} images, wrote {OUTPUT_JSON}")
