import os
import time
import pandas as pd
import argparse
import json
import csv
import logging
import gc

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations and paths
model_dir = "/model-weights/Llama-3.2-11B-Vision-Instruct"
data_folder = os.path.join("..", "..", "data", "eval2")
stratified_json = os.path.join(data_folder, "stratified_attribute_ids.json")
captions_json = os.path.join(data_folder, "eval2_processed_metadata.json")
prompt_txt = os.path.join(data_folder, "eval2_prompt.txt")
output_csv = os.path.join(data_folder, "eval2_llama_generated_questions.csv")
image_dir = os.path.join(data_folder, "eval2_processed_images")  # Directory where images are stored
offload_folder = "" # Folder to offload model weights for efficient memory usage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check that the image directory exists
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"The images directory '{image_dir}' does not exist. Please check the path.")

# Load model with 4-bit quantization configuration for efficient memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

logger.info("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=offload_folder
)
logger.info("Model loaded successfully.")

def load_prompt(prompt_filepath):
    """
    Load the main prompt from a text file.

    Args:
        prompt_filepath (str): Path to the prompt text file.

    Returns:
        str: The prompt text.
    
    Raises:
        FileNotFoundError: If the prompt file is not found.
        Exception: For other errors during file reading.
    """
    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"The prompt file '{prompt_filepath}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the prompt file: {e}")

def load_data(stratified_json_path, captions_json_path):
    """
    Load stratified attribute IDs and image captions metadata from JSON files.

    Args:
        stratified_json_path (str): Path to the stratified attributes JSON file.
        captions_json_path (str): Path to the captions metadata JSON file.

    Returns:
        tuple: (stratified_data, captions_data)

    Raises:
        ValueError: If JSON decoding fails.
    """
    try:
        with open(stratified_json_path, "r", encoding="utf-8") as f:
            stratified_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding stratified JSON file: {e}")

    try:
        with open(captions_json_path, "r", encoding="utf-8") as f:
            captions_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding captions JSON file: {e}")

    return stratified_data, captions_data

def resize_image(image, max_size):
    """
    Resize an image to fit within the given max_size while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The image to resize.
        max_size (tuple): Maximum (width, height).

    Returns:
        PIL.Image.Image: The resized image.
    """
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def preprocess_data(stratified_data, captions_data, processor, main_prompt):
    """
    Preprocess input data to prepare for question generation.

    For each attribute in the stratified data, find the corresponding image metadata 
    and create an input prompt that includes a social attribute and image description.
    Also load and resize the image.

    Args:
        stratified_data (dict): Dictionary mapping attributes to lists of IDs.
        captions_data (list): List of metadata dictionaries for images.
        processor (AutoProcessor): The processor for the model.
        main_prompt (str): The main prompt text.

    Returns:
        list: A list of tuples: (unique_id, attribute, image, input_text)
    """
    processed_data = []
    for attribute, ids in tqdm(stratified_data.items(), desc="Processing attributes"):
        for unique_id in ids:
            try:
                # Retrieve image metadata from captions data
                image_data = next((item for item in captions_data if item["id"] == unique_id), None)
                if image_data is None:
                    raise ValueError(f"ID {unique_id} not found in captions data.")

                image_path = os.path.join(image_dir, f"{unique_id}.jpg")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image for ID {unique_id} not found at {image_path}")

                image = Image.open(image_path).convert("RGB")
                description = image_data.get("image_description", "No description provided")
                social_attribute = image_data.get("attributes", "No attribute provided")

                # Construct the input prompt by combining main prompt, social attribute, and description
                prompt = f"{main_prompt}\n\nSocial Attribute: {social_attribute}\nDescription: {description}\n"
                input_text = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                    add_generation_prompt=True
                )
                processed_data.append((unique_id, attribute, image, input_text))
            except Exception as e:
                logger.error(f"Error processing ID {unique_id}: {e}")
    return processed_data

def generate_questions(processed_data, processor, model, device, output_csv_path):
    """
    Generate questions for each preprocessed entry and save results to a CSV file.

    For each tuple in the processed data, the image is resized, the input prompt is processed,
    and the model generates a response. The response is then post-processed to remove the prompt 
    and saved along with the ID and attribute.

    Args:
        processed_data (list): List of tuples (unique_id, attribute, image, input_text).
        processor (AutoProcessor): The processor for input preparation.
        model (AutoModelForImageTextToText): The model for generating responses.
        device (torch.device): Device to run the model on.
        output_csv_path (str): Path to the output CSV file.
    """
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["ID", "attribute", "generated_question"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for unique_id, attribute, image, input_text in tqdm(processed_data, desc="Generating questions"):
            try:
                # Resize image to a maximum size to save memory
                image = resize_image(image, max_size=(350, 350))

                # Prepare model inputs
                inputs = processor(
                    text=input_text,
                    images=image,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(device)

                # Generate the model response
                output = model.generate(**inputs, max_new_tokens=150)
                response = processor.decode(output[0], skip_special_tokens=True)

                # Remove everything before 'assistant' in the response, if present
                idx = response.find("assistant")
                if idx != -1:
                    response = response[idx:]
                writer.writerow({"ID": unique_id, "attribute": attribute, "generated_question": response})
            except Exception as e:
                logger.error(f"Error generating question for ID {unique_id}: {e}")
                writer.writerow({"ID": unique_id, "attribute": attribute, "generated_question": f"Error: {e}"})

if __name__ == "__main__":
    # Load the main prompt
    main_prompt = load_prompt(prompt_txt)
    
    # Load stratified attribute IDs and image captions metadata
    stratified_data, captions_data = load_data(stratified_json, captions_json)
    
    logger.info("Selected all items for evaluation.")
    total_items = sum(len(ids) for ids in stratified_data.values())
    logger.info(f"Total items: {total_items}")
    
    # Preprocess the data to prepare for generation
    processed_data = preprocess_data(stratified_data, captions_data, processor, main_prompt)
    
    # Generate questions for each processed entry and save to CSV
    generate_questions(processed_data, processor, model, device, output_csv)
    
    logger.info(f"Questions generated and saved to {output_csv}")
