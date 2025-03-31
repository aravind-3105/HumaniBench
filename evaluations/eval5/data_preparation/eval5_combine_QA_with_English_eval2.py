import os
import json

def combine_json_data(final_dir, selected_file, base_file, output_file, language):
    """
    Combine language-specific data with base English data.

    This function loads a language-specific JSON file (e.g., Bengali) and a base
    English JSON file, then combines entries by matching on 'ID' and 'Attribute'.
    If a match is found, it creates a combined entry that includes both English and
    language-specific questions and answers, and saves the result to an output file.

    Args:
        final_dir (str): Directory containing language JSON files (not used in this function).
        selected_file (str): Path to the language-specific JSON file.
        base_file (str): Path to the base English JSON file.
        output_file (str): Path where the combined JSON data will be saved.
        language (str): Language identifier for the language-specific data.
    """
    # Load language-specific data
    with open(selected_file, 'r', encoding='utf-8') as f:
        selected_data = json.load(f)
    print(f"Number of selected entries (language-specific) from {selected_file}: {len(selected_data)}")

    # Load base English data
    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    print(f"Number of entries in base file (English data) from {base_file}: {len(base_data)}")

    combined_data = []

    # Iterate over each entry in the base file
    for english_entry in base_data:
        entry_id = english_entry['ID']
        entry_attribute = english_entry['Attribute']

        # Search for a matching entry in the language-specific file
        matching_entry = None
        for language_entry in selected_data:
            if language_entry['ID'] == entry_id and language_entry['Attribute'] == entry_attribute:
                matching_entry = language_entry
                break
        
        # If a matching entry is found, combine the data
        if matching_entry:
            combined_entry = {
                "ID": entry_id,
                "Attribute": entry_attribute,
                "Question(English)": english_entry['Question'],
                "Answer(English)": english_entry['Answer'],
                "Language": language,
                f"Question({language})": matching_entry['Question'],
                f"Answer({language})": matching_entry['Answer']
            }
            combined_data.append(combined_entry)
        else:
            print(f"No match found for ID: {entry_id}, Attribute: {entry_attribute} in the selected file")

    # Save the combined data to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Combined data saved to {output_file}")


if __name__ == "__main__":
    final_dir = 'final_v2/'  # Directory containing language JSON files
    selected_file_set = "./data/OpenAI_Output/Eval2/final"
    selected_files = os.listdir(selected_file_set)
    base_file = "./data/Selected_QA_Eval2.json"  # English data file

    for selected_file in selected_files:
        language = selected_file.split('_')[1]  # Extract the language from the file name
        selected_file_path = os.path.join(selected_file_set, selected_file)
        output_file = f'final_v2/Eval2_{language}_combined.json'  # Output file to save the combined result
        combine_json_data(final_dir, selected_file_path, base_file, output_file, language)
