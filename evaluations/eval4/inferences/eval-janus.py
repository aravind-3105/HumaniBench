import os
import json
import re
import torch
from PIL import Image
from transformers import set_seed

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

# ---------------------------------------------------------------------------
# CONFIG
MODEL_NAME_OR_PATH = "deepseek-ai/deepseek-vl2-tiny"  # Your model on Hugging Face
LABELS_JSON = "/projects/NMB-Plus/E-VQA/results/eval4/yolo_detections.json"  # Ground-truth label file
IMAGES_DIR = "/projects/NMB-Plus/E-VQA/results/eval4/images"  # Directory with images
OUTPUT_JSON = "janus_bboxes_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = ""

# How many records to process? Set to an integer or None to process all.
NUM_RECORDS = None  # e.g., 10

# ---------------------------------------------------------------------------
# 1. Load ground-truth records
with open(LABELS_JSON, "r") as f:
    gt_records = json.load(f)

if NUM_RECORDS is not None:
    gt_records = gt_records[:NUM_RECORDS]

print(f"Loaded {len(gt_records)} records from {LABELS_JSON}")

# ---------------------------------------------------------------------------
# 2. Load model + processor
print(f"Loading model from {MODEL_NAME_OR_PATH}...")

model = DeepseekVLV2ForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).eval()
processor = DeepseekVLV2Processor.from_pretrained(
    MODEL_NAME_OR_PATH,
    cache_dir=CACHE_DIR,
)
tokenizer = processor.tokenizer
model.eval()

# ---------------------------------------------------------------------------
# 3. Prompt functions and helpers

def stage1_prompt(class_names):
    """
    Stage 1 prompt: Ask the model (in free-form) to detect objects.
    Uses special bounding-box tokens or reference tokens if needed.
    """
    prompt = (
        "<image>\n"
        "<|ref|>Detect the following classes: " + ", ".join(class_names) +
        ". Provide approximate bounding boxes and explain what you see.<|/ref|>"
    )
    return prompt

def stage2_refine_prompt(stage1_text, img_width, img_height):
    """
    Stage 2 prompt: Ask the model to output ONLY strict bounding box lines.
    Expected output format:
      Object: <class_name>, BBox: [x1, y1, x2, y2]
    """
    prompt = (
        "<|ref|>The previous output was:\n" + stage1_text +
        "\nNow, restate ONLY the bounding boxes in the exact format:\n"
        "Object: <class_name>, BBox: [x1, y1, x2, y2]\n"
        f"- If the coords were normalized, multiply by width ({img_width}) and height ({img_height}), then round.\n"
        "No extra text.<|/ref|>"
    )
    return prompt

def run_deepseek_inference(prompt_text, image=None):
    """
    Run inference with DeepSeek VL2 model using a proper conversation list,
    then convert the outputs to the format model.generate expects.
    """
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt_text,
            "images": [[image]] if image is not None else []
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        }
    ]

    # 1. Convert to a batch with the processor
    inputs = processor(
        conversations=conversation,
        images=[image] if image is not None else None,
        force_batchify=True,
        system_prompt="",
        return_tensors="pt"
    ).to(DEVICE)

    # 2. Convert these inputs into 'inputs_embeds' for the model
    #    This step removes 'sft_format' and 'seq_lens' from the arguments
    #    so they don't get passed to model.generate.
    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    # 3. Pass only the required arguments to model.generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id  # Ensure pad_token_id is passed
            # Optionally: pad_token_id, etc.
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_floats_bboxes(text):
    """
    Regex to capture lines like:
      Object: cat, BBox: [100, 40, 120, 80]
      Object: tie, BBox: [0.009, 0.373, 0.629, 0.986]
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
    Convert normalized coords [0..1] to pixel coords if values <= 1,
    else assume they're already pixel coords.
    Ensure x1 < x2 and y1 < y2.
    """
    x1, y1, x2, y2 = bbox
    if all(0 <= c <= 1.0 for c in (x1, y1, x2, y2)):
        x1 = round(x1 * img_width)
        y1 = round(y1 * img_height)
        x2 = round(x2 * img_width)
        y2 = round(y2 * img_height)
    else:
        x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [int(x1), int(y1), int(x2), int(y2)]

def iou(boxA, boxB):
    """
    Intersection over Union for boxes defined as [x1, y1, x2, y2].
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

# ---------------------------------------------------------------------------
# 4. Main Loop
predictions = []

for idx, record in enumerate(gt_records, start=1):
    filename = record["original_filename"]
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        print(f"[{idx}] Image not found: {image_path}, skipping.")
        continue

    annos = record.get("annotations", [])
    class_names = list({a["class_name"] for a in annos if "class_name" in a})
    if not class_names:
        print(f"[{idx}] No classes for {filename}, skipping.")
        continue

    img_width = record.get("img_width", 224)
    img_height = record.get("img_height", 224)

    print(f"[{idx}] Processing {filename} (classes={class_names})...")

    # Stage 1
    image_pil = Image.open(image_path).convert("RGB")
    prompt1 = stage1_prompt(class_names)
    stage1_out = run_deepseek_inference(prompt1, image=image_pil)

    # Stage 2
    prompt2 = stage2_refine_prompt(stage1_out, img_width, img_height)
    # Note: second stage does not strictly require the image, 
    #       but you *can* pass it if you want the model to see it again.
    stage2_out = run_deepseek_inference(prompt2, image=None)

    # Parse bounding boxes from stage 2 output
    parsed_boxes = parse_floats_bboxes(stage2_out)

    # Convert boxes to pixel coordinates and compute IoU with GT
    deepseek_boxes = []
    for p in parsed_boxes:
        px_box = convert_to_pixel(p["bbox"], img_width, img_height)
        deepseek_boxes.append({
            "class_name": p["class_name"],
            "bbox": px_box
        })

    # (Optional) Compute best IoU for each predicted box vs. ground truth
    for box_info in deepseek_boxes:
        best_iou_val = 0.0
        for gt in annos:
            if gt.get("class_name") == box_info["class_name"]:
                gt_box = list(map(int, map(round, gt["bbox"])))
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

    for box_info in deepseek_boxes:
        out_record["annotations"].append({
            "class_name": box_info["class_name"],
            "bbox": box_info["bbox"],
            "confidence": 1.0,  # Hard-coded confidence
            "best_iou_with_gt": round(box_info["best_iou"], 3)
        })

    predictions.append(out_record)

# ---------------------------------------------------------------------------
# 5. Write the final JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nDone! Processed {len(predictions)} images, wrote {OUTPUT_JSON}")
