import json
import os
import re

# Configuration for language-specific processing.
# For each language, we define:
# - remove_pattern: A regex pattern to remove from extracted question/answer.
# - alt_extraction: A tuple (question_marker, answer_marker) to use if the default extraction fails.
#   If answer_marker is None, only the question is extracted.
# - prefix_removals (optional): A list of prefixes to remove from the question.
LANGUAGE_CONFIG = {
    "Urdu": {
        "remove_pattern": r'\(in Urdu\)',
        "alt_extraction": None
    },
    "Tamil": {
        "remove_pattern": r'\(in Tamil\)',
        "alt_extraction": None
    },
    "Spanish": {
        "remove_pattern": r'\(in Spanish\)',
        "alt_extraction": ("Pregunta (en español):", "Respuesta (en español):")
    },
    "Punjabi": {
        "remove_pattern": r'\(in Punjabi\)',
        "alt_extraction": ("Pregunta (en español):", "Respuesta (en español):")
    },
    "Portuguese": {
        "remove_pattern": r'\(in Portuguese\)',
        "alt_extraction": ("Questão (em Português):", "Resposta (em Português):")
    },
    "Persian": {
        "remove_pattern": r'\(in Persian\)',
        "alt_extraction": ("پرسش (به فارسی):", "پاسخ (به فارسی):")
    },
    "Mandarin": {
        "remove_pattern": r'\(in Mandarin\)',
        "alt_extraction": ("问题（中文）：", None)
    },
    "Korean": {
        "remove_pattern": r'\(in Korean\)',
        "alt_extraction": ("질문 (한국어):", "답변 (한국어):")
    },
    "French": {
        "remove_pattern": r'\(in French\)',
        # For French, several alternative markers might be needed.
        "alt_extraction": None,
        "prefix_removals": ["Question (in French) :", "Question (in French) :  "]
    },
    "Bengali": {
        "remove_pattern": r'\(in Bengali\)',
        "alt_extraction": ("প্রশ্ন (বাংলায়):", None)
    }
}

def process_language(input_file, output_file, language):
    """
    Process an input JSON file to extract and clean question/answer pairs for a given language.
    
    This function loads the data from input_file (a JSON file containing a list of entries).
    Each entry should have a "Content" (or "content") field. It attempts to extract the question
    and answer by first splitting on the common markers "Question" and "Answer". Then it applies 
    language-specific cleaning (such as removing a pattern like "(in Urdu)") and, if extraction 
    fails, uses alternative markers defined in LANGUAGE_CONFIG.
    
    Finally, it cleans extra punctuation and whitespace before saving the processed entries to
    output_file.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
        language (str): Target language name (e.g. "Urdu", "Tamil", "Spanish", etc.).
    
    Returns:
        None
    """
    # Load configuration for the target language; if not defined, use default (empty config).
    config = LANGUAGE_CONFIG.get(language, {"remove_pattern": "", "alt_extraction": None})
    remove_pattern = config.get("remove_pattern", "")
    alt_extraction = config.get("alt_extraction", None)
    prefix_removals = config.get("prefix_removals", [])
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for entry in data:
        # Get the content from either "Content" or "content"
        content = entry.get('Content', '') or entry.get('content', '')
        question = None
        answer = None
        
        # Primary extraction: if "Question" is in content, split on it and then split on "Answer"
        if "Question" in content:
            extracted = content.split("Question", 1)[1].strip()
            if "Answer" in extracted:
                question, answer = extracted.split("Answer", 1)
                question = question.strip()
                answer = answer.strip()
            else:
                question = extracted
        
        # If no question was extracted (or it's empty), try alternative extraction if provided.
        if (not question or question == "") and alt_extraction:
            q_marker, a_marker = alt_extraction
            if q_marker in content:
                question = content.split(q_marker, 1)[1].strip()
                if a_marker and a_marker in question:
                    question, answer = question.split(a_marker, 1)
                    question = question.strip()
                    answer = answer.strip()
        
        # Remove language-specific pattern from question and answer, if they exist.
        if question:
            question = re.sub(remove_pattern, '', question).strip()
        if answer:
            answer = re.sub(remove_pattern, '', answer).strip()
        
        # Remove any prefixes specified in config (for example in French)
        for prefix in prefix_removals:
            if question and question.startswith(prefix):
                question = question.replace(prefix, '', 1).strip()
        
        # Clean up: remove trailing and leading colons and whitespace
        if question:
            question = question.strip().strip(':')
            # Remove extra quotes if present
            question = question.replace('\"', '').strip()
            # Remove any leading/trailing smart quotes
            question = question.strip('“”')
        if answer:
            answer = answer.strip().strip(':')
            answer = answer.replace('\"', '').strip()
            answer = answer.strip('“”')
        
        # Prepare processed entry
        processed_entry = {
            "ID": entry.get("ID"),
            "Attribute": entry.get("Attribute"),
            "Content": content,
            "Question": question,
            "Answer": answer
        }
        processed_data.append(processed_entry)
    
    # Save processed data to output file in UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    # For demonstration, you could process one file per language.
    # Adjust the filenames as needed.
    language_files = {
        "Urdu": ("results_Urdu_processed.json", "results_Urdu_processed_output.json"),
        "Tamil": ("results_Tamil_processed.json", "results_Tamil_processed_output.json"),
        "Spanish": ("results_Spanish_processed.json", "results_Spanish_processed_output.json"),
        "Punjabi": ("results_Punjabi_processed.json", "results_Punjabi_processed_output.json"),
        "Portuguese": ("results_Portuguese_processed.json", "results_Portuguese_processed_output.json"),
        "Persian": ("results_Persian_processed.json", "results_Persian_processed_output.json"),
        "Mandarin": ("results_Mandarin_processed.json", "results_Mandarin_processed_output.json"),
        "Korean": ("results_Korean_processed.json", "results_Korean_processed_output.json"),
        "French": ("results_French_processed.json", "results_French_processed_output.json"),
        "Bengali": ("results_Bengali_processed.json", "results_Bengali_processed_output.json")
    }
    
    # Process each language file using the unified function.
    for lang, (input_file, output_file) in language_files.items():
        print(f"Processing {lang}...")
        process_language(input_file, output_file, lang)
        print("---")