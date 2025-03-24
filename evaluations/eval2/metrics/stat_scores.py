import os
import time
import pandas as pd
import argparse
import evaluate
from tqdm import tqdm
import json
import gc

def load_metrics():
    """
    Load evaluation metrics once.

    Returns:
        tuple: A tuple containing the loaded FrugalScore, BERTScore, and METEOR metrics.
    """
    frugalscore = evaluate.load("frugalscore", "moussaKam/frugalscore_medium_bert-base_mover-score")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    return frugalscore, bertscore, meteor

def process_batch(data_batch, frugalscore, bertscore, meteor):
    """
    Process a batch of data to compute evaluation metrics.

    For each row in the batch DataFrame, it concatenates the model answer and reasoning
    into one prediction, and then computes FrugalScore, BERTScore (using a specified model),
    and METEOR against the provided ground truth.

    Args:
        data_batch (DataFrame): A batch of data.
        frugalscore: Loaded FrugalScore metric.
        bertscore: Loaded BERTScore metric.
        meteor: Loaded METEOR metric.

    Returns:
        tuple: A tuple containing the results from FrugalScore, BERTScore, and METEOR.
    """
    predictions = []
    references = []

    for i, row in data_batch.iterrows():
        answer = row['Model_Answer'] if pd.notnull(row['Model_Answer']) else ""
        reasoning = row['Model_Reasoning'] if pd.notnull(row['Model_Reasoning']) else ""
        prediction = answer + " " + reasoning
        predictions.append(prediction)
        references.append(row['Ground_Truth'])
    
    results_frugalscore = frugalscore.compute(predictions=predictions, references=references)
    results_bertscore = bertscore.compute(predictions=predictions, 
                                          references=references, 
                                          model_type='microsoft/deberta-xlarge-mnli')
    results_meteor = meteor.compute(predictions=predictions, references=references)
    
    return results_frugalscore, results_bertscore, results_meteor

def save_results(scores, output_path):
    """
    Save the computed scores to a JSON file.

    Args:
        scores (dict): Dictionary containing evaluation scores.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=4)

def main(input_path, output_path=None, batch_size=1000):
    """
    Evaluate model predictions in a CSV file and compute evaluation metrics in batches.

    This function loads the input CSV file, resumes from the last processed row (if output
    file exists), and processes the data in batches. For each batch, it computes FrugalScore,
    BERTScore, and METEOR, accumulates individual scores for averaging, and finally saves
    the averaged scores to the specified output file.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to the output JSON file (if provided).
        batch_size (int): Number of rows to process per batch.
    """
    # Load data from CSV
    data = pd.read_csv(input_path)
    print(f"Data size: {len(data)}")
    
    # Load evaluation metrics once
    frugalscore, bertscore, meteor = load_metrics()

    # Initialize lists for averaging BERTScore and FrugalScore components
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    frugal_scores = []

    scores = {
        "FrugalScore": [],
        "BERTScore": [],
        "METEOR": []
    }

    # Process the data in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        data_batch = data.iloc[i:i + batch_size]
        results_frugalscore, results_bertscore, results_meteor = process_batch(
            data_batch, frugalscore, bertscore, meteor
        )
        
        scores["FrugalScore"].append(results_frugalscore)
        scores["BERTScore"].append(results_bertscore)
        scores["METEOR"].append(results_meteor)

        bert_precision.extend(results_bertscore["precision"])
        bert_recall.extend(results_bertscore["recall"])
        bert_f1.extend(results_bertscore["f1"])
        frugal_scores.extend(results_frugalscore["scores"])

        del data_batch, results_frugalscore, results_bertscore, results_meteor
        gc.collect()

    # Compute averages for BERTScore and FrugalScore
    scores["BERTScore_avg_precision"] = sum(bert_precision) / len(bert_precision) if bert_precision else 0
    scores["BERTScore_avg_f1"] = sum(bert_f1) / len(bert_f1) if bert_f1 else 0
    scores["BERTScore_avg_recall"] = sum(bert_recall) / len(bert_recall) if bert_recall else 0
    scores["FrugalScore_avg_scores"] = sum(frugal_scores) / len(frugal_scores) if frugal_scores else 0

    # Remove individual batch results from scores
    del scores["BERTScore"]
    del scores["FrugalScore"]

    # Save final results if an output path is provided
    if output_path:
        save_results(scores, output_path)
    
    return

if __name__ == "__main__":
    parent_folder = ""  # Adjust if needed
    folder = os.path.join(parent_folder, "eval2_cleaned_csv")
    output_folder = os.path.join(parent_folder, "eval2_statistical_scores")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process only CSV files in the folder
    files = [file for file in os.listdir(folder) if file.endswith(".csv")]
    
    for file in files:
        print(f"Processing file: {file}")
        output_file_name = os.path.splitext(file)[0]
        file_path = os.path.join(folder, file)
        output_file_path = os.path.join(output_folder, output_file_name + ".json")
        main(file_path, output_file_path)
        time.sleep(3)
