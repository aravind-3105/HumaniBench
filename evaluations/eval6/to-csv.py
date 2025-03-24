import json
import csv
import argparse

def load_json_data(json_filename):
    """Load data from a JSON file."""
    with open(json_filename, "r", encoding="utf-8") as file:
        return json.load(file)

def write_to_csv(data, csv_filename, fieldnames):
    """Write data to a CSV file."""
    with open(csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            row = {
                "id": item.get("id", ""),
                "image_path": item.get("image_path", ""),
                "csv_simple": item.get("csv_captions", {}).get("csv_simple", ""),
                "csv_emphatic": item.get("csv_captions", {}).get("csv_emphatic", ""),
                "model_simple": item.get("model_captions", {}).get("model_simple", ""),
                "model_empathetic": item.get("model_captions", {}).get("model_empathetic", ""),
            }
            writer.writerow(row)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert JSON data to CSV format.")
    parser.add_argument("json_filename", help="Path to the input JSON file")
    parser.add_argument("csv_filename", help="Path to the output CSV file")
    
    args = parser.parse_args()
    
    # Define column names for CSV
    fieldnames = [
        "id", 
        "image_path", 
        "csv_simple", 
        "csv_emphatic", 
        "model_simple", 
        "model_empathetic"
    ]
    
    # Load the JSON data and write it to a CSV file
    data = load_json_data(args.json_filename)
    write_to_csv(data, args.csv_filename, fieldnames)

    print(f"CSV file created at: {args.csv_filename}")

if __name__ == "__main__":
    main()
