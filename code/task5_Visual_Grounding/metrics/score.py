import json
import argparse
import os

def evaluate_bbox_predictions(input_file, output_file, iou_threshold=0.2):
    with open(input_file, "r") as f:
        data = json.load(f)

    TP, FP, FN = 0, 0, 0
    ious = []
    per_class_stats = {}

    for item in data:
        gt_boxes = item.get("ground_truth_annotations", [])
        pred_boxes = item.get("annotations", [])

        matched_preds = [pred.get("best_iou_with_gt", 0.0) >= iou_threshold for pred in pred_boxes]
        matched_count = sum(matched_preds)

        for pred in pred_boxes:
            iou = pred.get("best_iou_with_gt", 0.0)
            class_name = pred.get("class_name", "unknown")

            if iou >= iou_threshold:
                TP += 1
            else:
                FP += 1

            ious.append(iou)

            if class_name not in per_class_stats:
                per_class_stats[class_name] = {"TP": 0, "FP": 0, "FN": 0, "ious": []}

            if iou >= iou_threshold:
                per_class_stats[class_name]["TP"] += 1
            else:
                per_class_stats[class_name]["FP"] += 1
            per_class_stats[class_name]["ious"].append(iou)

        FN += max(0, len(gt_boxes) - matched_count)

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

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    average_iou = sum(ious) / len(ious) if ious else 0.0

    print("=== OVERALL METRICS ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"Average IoU: {average_iou:.3f}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}\n")

    print("=== PER-CLASS METRICS ===")
    for class_name, stats in per_class_stats.items():
        p = stats["TP"] / (stats["TP"] + stats["FP"]) if stats["TP"] + stats["FP"] > 0 else 0.0
        r = stats["TP"] / (stats["TP"] + stats["FN"]) if stats["TP"] + stats["FN"] > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        avg_iou = sum(stats["ious"]) / len(stats["ious"]) if stats["ious"] else 0.0

        print(f"Class: {class_name}")
        print(f"  Precision: {p:.3f}")
        print(f"  Recall:    {r:.3f}")
        print(f"  F1-score:  {f:.3f}")
        print(f"  Average IoU: {avg_iou:.3f}")
        print(f"  TP: {stats['TP']}, FP: {stats['FP']}, FN: {stats['FN']}")

    # Save results
    results = {
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA Phi4 bounding box predictions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file with predictions.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--iou_threshold", type=float, default=0.2, help="IoU threshold to consider a prediction as TP.")
    args = parser.parse_args()

    evaluate_bbox_predictions(args.input_file, args.output_file, args.iou_threshold)
