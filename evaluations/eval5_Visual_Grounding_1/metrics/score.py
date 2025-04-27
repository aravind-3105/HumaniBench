import json

# Load the file
with open("llava_phi4_bboxes.json", "r") as f:
    data = json.load(f)

# Settings
iou_threshold = 0.2
TP = 0
FP = 0
FN = 0
ious = []

# For per-class stats
per_class_stats = {}

for item in data:
    gt_boxes = item.get("ground_truth_annotations", [])
    pred_boxes = item.get("annotations", [])

    # Track matched GTs (to compute FN)
    matched_preds = [pred.get("best_iou_with_gt", 0.0) >= iou_threshold for pred in pred_boxes]
    matched_count = sum(matched_preds)

    # Count per prediction (TP or FP)
    for pred in pred_boxes:
        iou = pred.get("best_iou_with_gt", 0.0)
        class_name = pred.get("class_name", "unknown")

        if iou >= iou_threshold:
            TP += 1
        else:
            FP += 1

        ious.append(iou)

        # Per-class stats init
        if class_name not in per_class_stats:
            per_class_stats[class_name] = {"TP": 0, "FP": 0, "FN": 0, "ious": []}

        # Update per-class TP/FP
        if iou >= iou_threshold:
            per_class_stats[class_name]["TP"] += 1
        else:
            per_class_stats[class_name]["FP"] += 1
        per_class_stats[class_name]["ious"].append(iou)

    # Count FN globally
    FN += max(0, len(gt_boxes) - matched_count)

    # Count FN per class
    gt_class_count = {}
    for gt in gt_boxes:
        cname = gt.get("class_name", "unknown")
        gt_class_count[cname] = gt_class_count.get(cname, 0) + 1

    for cname, count in gt_class_count.items():
        matched_c = sum(
            1 for pred in pred_boxes
            if pred.get("class_name") == cname and pred.get("best_iou_with_gt", 0.0) >= iou_threshold
        )
        if cname not in per_class_stats:
            per_class_stats[cname] = {"TP": 0, "FP": 0, "FN": 0, "ious": []}
        per_class_stats[cname]["FN"] += max(0, count - matched_c)

# Final overall metrics
precision = TP / (TP + FP) if TP + FP > 0 else 0.0
recall = TP / (TP + FN) if TP + FN > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
average_iou = sum(ious) / len(ious) if ious else 0.0

print("=== OVERALL METRICS ===")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Average IoU: {average_iou:.3f}")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print("\n=== PER-CLASS METRICS ===")
for class_name, stats in per_class_stats.items():
    precision = stats["TP"] / (stats["TP"] + stats["FP"]) if stats["TP"] + stats["FP"] > 0 else 0.0
    recall = stats["TP"] / (stats["TP"] + stats["FN"]) if stats["TP"] + stats["FN"] > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    average_iou = sum(stats["ious"]) / len(stats["ious"]) if stats["ious"] else 0.0

    print(f"Class: {class_name}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print(f"  Average IoU: {average_iou:.3f}")
    print(f"  TP: {stats['TP']}, FP: {stats['FP']}, FN: {stats['FN']}")
# Save the results to a JSON file
output_data = {
    "overall": {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_iou": average_iou,
        "TP": TP,
        "FP": FP,
        "FN": FN
    },
    "per_class": per_class_stats
}
with open("./results/evaluation_results.json", "w") as f:
    json.dump(output_data, f, indent=4)