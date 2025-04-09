import pandas as pd
from bert_score import score
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Consolidation of Results")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file")
    #parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()
    
    # Load your CSVs
    ground_truth_csv = "./data/eval1/gpt4o-plain_version_cleaned.csv"
    predicted_csv    = args.input
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df_gt = pd.read_csv(ground_truth_csv,low_memory=False)
    df_pred = pd.read_csv(predicted_csv,low_memory=False)
    
    # Merge on common column "id"
    df_merged = df_gt.merge(df_pred, on="id", suffixes=("_gt", "_pred"))
    
    # Retrieve the responses
    refs = df_merged["response_gt"].fillna("").tolist()
    hyps = df_merged["response_pred"].fillna("").tolist()
    
    # Calculate BERTScore while forcing the model to use CPU.
    # The `device="cpu"` parameter ensures that the model is not moved to GPU.
    P, R, F1 = score(hyps, refs, lang="en", verbose=True, device=device)
    
    # Add scores to dataframe
    df_merged["BERTScore_P"] = P.tolist()
    df_merged["BERTScore_R"] = R.tolist()
    df_merged["BERTScore_F1"] = F1.tolist()
    
    # Compute the average F1 score across all rows as a semantic similarity measure.
    average_F1 = df_merged["BERTScore_F1"].mean()
    print(f"Average BERTScore F1: {average_F1:.3f}")
    
    # Optionally, save the merged results with scores for further analysis.
    df_merged.to_csv("merged_with_bert_score.csv", index=False)
    # print("Saved merged results with BERTScore to merged_with_bert_score.csv")

if __name__ == "__main__":
    main()
