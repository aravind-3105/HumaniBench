import os
import json
import pandas as pd

def convert_json_to_csv(input_folder, output_folder):
    """
    Convert all JSON files in the input folder to CSV files saved in the output folder.

    Each JSON file should be a list of dictionaries with the columns:
    "ID", "Question", "Predicted_Answer", "Model_Answer", "Model_Reasoning", "Ground_Truth", "Attribute".
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(input_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Processing {file}: {len(data)} entries found.")
            df = pd.DataFrame(data, columns=[
                "ID", "Question", "Predicted_Answer",
                "Model_Answer", "Model_Reasoning",
                "Ground_Truth", "Attribute"
            ])
            output_file = os.path.join(output_folder, file.replace(".json", ".csv"))
            df.to_csv(output_file, index=False)
            print(f"Saved: {file.replace('.json', '.csv')}")

if __name__ == "__main__":
    parent_folder = ""  # Adjust if necessary
    input_folder_path = os.path.join(parent_folder, "eval2_cleaned")
    output_folder = os.path.join(parent_folder, "eval2_cleaned_csv")
    convert_json_to_csv(input_folder_path, output_folder)
