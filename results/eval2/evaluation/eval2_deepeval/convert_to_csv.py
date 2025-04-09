import os
import json
import pandas as pd
import time


# file_path = "results_gemma3_12b_cleaned.json"

# COnvet json to csv
# Headers are headers = ["ID", "Question", "Predicted_Answer", "Model_Answer", "Model_Reasoning", "Ground_Truth", "Attribute"]
input_folder_path = "/projects/NMB-Plus/E-VQA/source/eval2/evaluation/eval2_cleaned"
files = os.listdir(input_folder_path)
# Make folder to save csv
folder = "/projects/NMB-Plus/E-VQA/source/eval2/evaluation/eval2_cleaned_csv"
# Check if folder exists
if not os.path.exists(folder):
    os.makedirs(folder)
for file in files:
    if file.endswith(".json"):
        file_path = os.path.join(input_folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
    # Print length of data
    print(len(data))
    df = pd.DataFrame(data, columns=["ID", "Question", "Predicted_Answer", "Model_Answer", "Model_Reasoning", "Ground_Truth", "Attribute"])
    # Save to csv
    df.to_csv(os.path.join(folder, file.replace(".json", ".csv")), index=False)
    print("Saved: ", file.replace(".json", ".csv"))



