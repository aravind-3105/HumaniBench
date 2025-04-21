import json
import os
import glob

def process_jsonl_files(jsonl_files, output_json_file):
    """
    Process a list of JSONL files and combine their entries into a single JSON file.

    Each line in a JSONL file should be a JSON object containing a "custom_id" field,
    which is split into "ID" and "Attribute", and a "response" field from which content
    is extracted.

    Args:
        jsonl_files (list): List of paths to JSONL files to be processed.
        output_json_file (str): Path to the output JSON file where results are saved.

    Returns:
        None. The combined results are written to the output JSON file.
    """
    results = []

    # Iterate over each JSONL file in the provided list
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    entry = json.loads(line.strip())

                    # Extract custom_id and split it into ID and Attribute
                    custom_id = entry.get("custom_id", "")
                    if custom_id:
                        id_part, attribute_part = custom_id.split("_", 1)
                    else:
                        id_part, attribute_part = "", ""

                    # Extract content from the response if available
                    content = ""
                    if entry.get("response") and entry["response"].get("body"):
                        body = entry["response"]["body"]
                        choices = body.get("choices", [{}])
                        content = choices[0].get("message", {}).get("content", "")

                    # Append the processed entry to the results list
                    results.append({
                        "ID": id_part,
                        "Attribute": attribute_part,
                        "content": content
                    })
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line}")
                except Exception as e:
                    print(f"An error occurred while processing the file: {e}")

    # Write the combined results to the output JSON file
    with open(output_json_file, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_json_file}")

if __name__ == "__main__":
    # Get all .jsonl files in the current directory
    jsonl_files = glob.glob("*.jsonl")
    # Create output file names for each JSONL file (e.g., input.jsonl -> input_processed.json)
    output_files = [f"{os.path.splitext(file)[0]}_processed.json" for file in jsonl_files]

    # Process each JSONL file individually
    for input_file, output_file in zip(jsonl_files, output_files):
        process_jsonl_files([input_file], output_file)
