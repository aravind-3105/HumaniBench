import json
import re
import os

# Common processing logic for all models
def process_data(data, model_config):
    """
    Process the data according to model-specific rules.
    Args:
        data (list): List of dictionaries containing the data to be processed.
        model_config (dict): Dictionary containing model-specific processing rules.
    Returns:
        list: List of dictionaries containing the cleaned data.
    """
    results = []
    for entry in data:
        predicted_answer = entry['Predicted_Answer']

        # Apply model-specific processing rules
        if model_config.get("reasoning_split"):
            reasoning = None
            if model_config["reasoning_split"] in predicted_answer:
                predicted_answer, reasoning = predicted_answer.split(model_config["reasoning_split"])
            reasoning = reasoning.strip() if reasoning else None
        
        # Apply answer extraction rule
        if model_config.get("answer_split"):
            predicted_answer = predicted_answer.split(model_config["answer_split"])[-1]
        
        # Apply any additional text clean up
        predicted_answer = clean_text(predicted_answer, model_config.get("cleanup_tags"))
        if reasoning:
            reasoning = clean_text(reasoning, model_config.get("cleanup_tags"))

        # Add cleaned data to results
        results.append({
            'ID': entry['ID'],
            'Question': entry['Question'],
            'Predicted_Answer': entry['Predicted_Answer'],
            'Model_Answer': predicted_answer.strip(),
            'Model_Reasoning': reasoning,
            'Ground_Truth': entry['Ground_Truth'],
            'Attribute': entry['Attribute']
        })

    return results

# Helper function to clean the text by removing specified tags
def clean_text(text, tags=None):
    """
    Clean the text by removing specified tags and whitespace.
    Args:
        text (str): The text to be cleaned.
        tags (list): List of regex patterns to remove from the text.
    Returns:
        str: The cleaned text.
    """
    if not text:
        return None
    if tags:
        for tag in tags:
            text = re.sub(tag, '', text)
    return text.strip()

# Model configurations: Rules to apply for each model
MODEL_CONFIGS = {
    'Llama_Vision': {
        "reasoning_split": "Reasoning:",
        "answer_split": 'Answer:',
        "cleanup_tags": [r'<reasoning>', r'<answer>', r'[<>]']
    },
    'Phi': {
        "reasoning_split": "Reasoning:",
        "answer_split": 'Answer:',
        "cleanup_tags": [r'<reasoning>', r'<answer>', r'[<>]']
    },
    'Aya': {
        "reasoning_split": "Reasoning:",
        "answer_split": 'Answer:',
        "cleanup_tags": [r'<reasoning>', r'<answer>', r'[<>]']
    },
    'gemma3_12b': {
        "reasoning_split": "Reasoning:",
        "answer_split": 'Answer:',
        "cleanup_tags": [r'<reasoning>', r'<answer>', r'[<>]']
    }
}

# Main function to process files
def process_files(RESULTS_FOLDER, SAVE_FOLDER):
    """
    Process JSON files in the specified folder, clean the data according to model-specific rules,
    and save the cleaned data to a new folder.

    Args:
        RESULTS_FOLDER (str): Path to the folder containing the JSON files to be processed.
        SAVE_FOLDER (str): Path to the folder where cleaned JSON files will be saved.
    """
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]

    for file in files:
        cleaned_file_path = os.path.join(SAVE_FOLDER, file.replace(".json", "_cleaned.json"))
        with open(os.path.join(RESULTS_FOLDER, file)) as f:
            data = json.load(f)

        # Identify model from the filename
        model_name = next((model for model in MODEL_CONFIGS if model in file), None)
        if model_name:
            model_config = MODEL_CONFIGS[model_name]
            cleaned_data = process_data(data, model_config)
            # Save cleaned data
            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=4)
            print(f"File {file} cleaned and saved to {cleaned_file_path}")
        else:
            print(f"File {file} not processed due to unknown model")

if __name__ == "__main__":
    RESULTS_FOLDER = "./results"
    SAVE_FOLDER = "./results/cleaned"
    process_files(RESULTS_FOLDER, SAVE_FOLDER)

# To run the script:
# python postprocess.py