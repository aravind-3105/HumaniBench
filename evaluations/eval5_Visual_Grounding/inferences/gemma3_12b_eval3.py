import torch
import json
import os
import time
import logging
from argparse import ArgumentParser
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.image_utils import load_image

# Model directory
MODEL_DIR = "/model-weights/gemma3-12b-it"  # Local model path
HF_MODEL_ID = "google/gemma-3-12b-it"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_source="local"):
    model_path = MODEL_DIR if model_source == "local" else HF_MODEL_ID
    logger.info(f"Loading Gemma3 model from {model_path}...")

    # os.environ["TRANSFORMERS_CACHE"] = "/scratch/ssd004/scratch/aravindn/huggingface_cache"

        # Set Hugging Face cache directory
    os.environ["HF_HOME"] = ""
    os.environ["TRANSFORMERS_CACHE"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype=torch.bfloat16,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    return model, processor

# Resize image
def resize_image(img_path, max_size=(350, 350)):
    try:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail(max_size, Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image {img_path}: {e}")
        return None

def extract_answer_and_reason(text):
    """Extracts the answer and reason from the VLM response"""

    try:
        # Find the first '{' and the last '}' to extract a JSON string.
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            answer = data.get("Answer", "").strip()
            reasoning = data.get("Reasoning", "").strip()
            if answer and reasoning:
                return answer, reasoning
            
    except Exception as e:
        pass
    
    # If not JSON then extract the text directly using regex
    try:
        pattern = r'(?:\*\*Answer:\*\*|Answer:)\s*"?([^"\n]*)"?\s*(?:\*\*Reasoning:\*\*|Reasoning:)\s*"?([^"\n]*)"?'
    
        match = re.search(pattern, text)

        if match:
            answer = match.group(1)
            reasoning = match.group(2)
            return answer, reasoning
        else:
            return text, None
    except:
        return text, None




# Process a single image-question pair and generate an answer
def process_sample(model, processor, img_path, question,language):
    try:
        # Load & resize the image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image = resize_image(img_path)
        if image is None:
            return "Error: Could not process image"
        
        # user_prompt = f"Given question, answer in the following format:\
        #         Question:{question}\
        #         Answer:<answer> \
        #         Reasoning:<reasoning> in the context of the image."

        # Format the prompt with the provided language
        query_prompt= f"""Answer the question using one of the given choices based on the image.

        Question ({language}):
        {question}

        Provide the response only in the following JSON format:
        {{"Answer": "The correct letter and option",
        "Reasoning": "A brief explanation (max 80 words) based on the details in the image"}}

        Do not provide any other extra information.
        """

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant following user's instructions. Don't add anything extra."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query_prompt}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        # print(decoded)

        return decoded if decoded != "" else "Error"


    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return "Error"



# Main function to process dataset
def evaluate(model, processor, dataset, image_folder, save_path, language, mode="single"):
    results = []
    logger.info(f"Starting evaluation in {mode} mode...")
    intermediate_results_path = save_path.replace(".json", "_intermediate.json")
    prev_path = ""
    with tqdm(total=len(dataset), unit="sample") as pbar:
        for i, data in enumerate(dataset):
            # print(data)
            img_path = os.path.join(image_folder, f"{data['ID']}.jpg")

            q_key = f"Question({language})"  # Dynamically choose the key
            o_key = f"Options({language})"  # Dynamically choose the key
            a_key = f"Answer({language})"  # Dynamically choose the key
            r_key = f"Reasoning({language})"  # Dynamically choose the key

            # Check if the keys exist in the dataset
            if q_key not in data or a_key not in data:
                logger.warning(f"Missing keys for {language} in dataset entry {data['ID']}")
                continue
            # Check if data[q_key] or data[o_key] is null
            if not data[q_key] or not data[o_key]:
                logger.warning(f"Empty question or choices for {language} in dataset entry {data['ID']}")
                results.append({
                    "ID": data["ID"],
                    "Question": data[q_key],
                    "Predicted_Answer": None,
                    "Predicted_Reasoning": None,
                    "Ground_Truth_Answer": data[a_key],
                    "Ground_Truth_Reasoning": data[r_key],
                    "Attribute": data["Attribute"]
                })
                pbar.update(1)
                continue


            question_with_choices = data[q_key] + '\n' + data[o_key]


            # Process the image and question
            answer = process_sample(model, processor, img_path, question_with_choices, language)
            # Extract answer and reasoning
            answer, reason = extract_answer_and_reason(answer)
            # Append results
            results.append({
                "ID": data["ID"],
                "Question": question_with_choices,
                "Predicted_Answer": answer,
                "Predicted_Reasoning": reason,
                "Ground_Truth_Answer": data[a_key],
                "Ground_Truth_Reasoning": data[r_key],
                "Attribute": data["Attribute"],
            })


            # Save intermediate results every 50
            if i % 10 == 0:
                intermediate_results_path = save_path.replace(".json", f"_intermediate_{i}_{time.strftime('%Y%m%d_%H%M%S')}.json")
                with open(intermediate_results_path, "w") as f:
                    json.dump(results, f, indent=4, default=str, ensure_ascii=False)
                if prev_path != "":
                    os.remove(prev_path)  
                logger.info(f"Intermediate results saved to {intermediate_results_path} and deleted {prev_path}.")
                prev_path = intermediate_results_path

            pbar.update(1)

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str, ensure_ascii=False)
    
    logger.info(f"Results saved to {save_path}.")



# Main execution
if __name__ == "__main__":
    start_time = time.time()

    # Command-line arguments
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data/eval5/eval3/Eval3_French.json", help="Path to dataset")
    parser.add_argument("--image_folder", type=str, default="./data/processed_images", help="Path to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--save_path", type=str, default="./results/results_Gemma3_Eval3_French.json", help="Output file to save results")
    parser.add_argument("--model_source", type=str, default="local", help="Model source: 'local' or 'hf'")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch"], help="Single or batch processing")


    args = parser.parse_args()
    
    # Define device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    language = args.dataset.split("_")[-1].split(".")[0]
    logger.info(f"Processing dataset in {language} language with {args.mode} mode...")


    model, processor = load_model(args.model_source)
    model.to(device)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Run evaluation
    evaluate(model, processor, dataset, args.image_folder, args.save_path, language, args.mode)

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
