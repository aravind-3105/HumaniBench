#!/usr/bin/env python
import os
import json
import csv
import torch
from PIL import Image
from argparse import ArgumentParser
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig
)

# Set environment variables for caching
os.environ["HF_HOME"] = "" #Path where you want to store the huggingface cache
os.environ["TRANSFORMERS_CACHE"] = "" #Path where you want to store the transformers cache


def main():
    parser = ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--phi4_model", type=str, default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results")
    os.makedirs(args.results_folder, exist_ok=True)
    parser.add_argument("--results_file", type=str, default="phi4_results.json")
    parser.add_argument("--image_folder", type=str, required=True)
    args = parser.parse_args()

    # Log in to Hugging Face
    login(token=args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Phi-4 multimodal components
    processor = AutoProcessor.from_pretrained(
        args.phi4_model,
        cache_dir=os.environ["HF_HOME"],
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.phi4_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        cache_dir=os.environ["HF_HOME"],
        _attn_implementation='eager'
    ).eval()

    # Load CSV data
    csv_data = {}
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = os.path.splitext(row["Image"])[0]
            csv_data[image_id] = {
                "simple": row["simple_Caption"],
                "emphatic": row["emphatic_Caption"]
            }

    results = []
    for image_id, captions in csv_data.items():
        print(f"Processing {image_id}...")
        image_path = os.path.join(args.image_folder, f"{image_id}.jpg")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_id}: {str(e)}")
            continue

        # Define prompts using correct template
        prompts = {
            "simple": "<|user|><|image_1|>Describe this image concisely.<|end|><|assistant|>",
            "emphatic": "<|user|><|image_1|>Create an empathetic caption for this image considering the emotional context.<|end|><|assistant|>"
        }

        result_entry = {
            "id": image_id,
            "csv_captions": captions,
            "phi4_captions": {}
        }

        for caption_type, prompt in prompts.items():
            # Format input according to Phi-4 requirements
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device)

            # Generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Decode response properly
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            result_entry["phi4_captions"][caption_type] = response

        results.append(result_entry)

        # Save after each image to prevent data loss
        results_file_path = os.path.join(args.results_folder, args.results_file)
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Completed processing {len(results)} images")

if __name__ == "__main__":
    main()