import os
import json
import argparse

def count_json_lists_in_directory(directory):
    json_list_lengths = {}

    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                        # Check if the data is a list
                        if isinstance(data, list):
                            json_list_lengths[file_path] = len(data)

                        # If the data is a dictionary, check for lists within it
                        elif isinstance(data, dict):
                            for key, value in data.items():
                                if isinstance(value, list):
                                    json_list_lengths[f"{file_path} -> {key}"] = len(value)

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {file_path}: {e}")

    return json_list_lengths

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Count JSON list lengths in a directory.')
    parser.add_argument('--dir', type=str, help='Path to the directory containing JSON files.')
    
    args = parser.parse_args()
    lengths = count_json_lists_in_directory(args.dir)

    num_sample_summary = {
        'gsm': 0,
        'math': 0,
    }
    for file, length in lengths.items():
        print(f"{file}: {length}")
        for key in num_sample_summary.keys():
            if key in file:
                num_sample_summary[key] += length

    print("===== summary ====")
    for key, num_sample in num_sample_summary.items():
        print(f"{key}: {num_sample}")

