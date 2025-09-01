
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import json
import re
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Sample JSON dataset
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    """The higher the threshold, the more similar the texts have to be to match"""

    # 预过滤长度小于等于1的文本，避免后续多次判断
    valid_indices = [i for i, item in enumerate(json_data) if item.get(key) and len(item[key]) > 1]
    text = [preprocess_text(json_data[i][key]) for i in valid_indices]
    if not text:
        return {}, []

    import numpy as np
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    near_duplicates = []
    indices_to_remove = set()
    n = len(valid_indices)
    # 只遍历有效索引，避免无效文本
    for i in range(n):
        for j in range(i+1, n):
            if cos_sim_matrix[i, j] > threshold:
                idx_i, idx_j = valid_indices[i], valid_indices[j]
                near_duplicates.append((json_data[idx_i], json_data[idx_j], cos_sim_matrix[i, j]))
                if key in ("input", "output"):
                    indices_to_remove.add(idx_j)

    filtered_json_data = [item for idx, item in enumerate(json_data) if idx not in indices_to_remove]
    return filtered_json_data, near_duplicates


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.
    """
    for key in json_data[0].keys():

        if remove_duplicates:
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50 * '='
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    return json_data


if __name__ == "__main__":
    print("scikit-learn version:", sklearn_version)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help=("Path to the dataset JSON file")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=("A sensitivity threshold between 0 and 1 where 1 is strictest")
    )
    parser.add_argument(
        "--remove_duplicates",
        action='store_true',
        default=False,
        help=(
            "Removes duplicates based on the 'input' or 'output' keys "
            " (but not the 'instruction') and saves the cleaned JSON file as --json_output_file"
        )
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help=("Path to the dataset JSON file")
    )

    args = parser.parse_args()

    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "Provide an output file via --json_output_file "
            "to save the cleaned JSON data."
        )

    if not args.json_file:
        json_data = example_data

    else:
        with open(args.json_file, "r") as file:
            json_data = json.load(file)

    json_data = find_print_and_remove_near_duplicates(
        json_data=json_data,
        remove_duplicates=args.remove_duplicates,
        threshold=args.threshold
    )

    if args.remove_duplicates:
        with open(args.json_output_file, "w") as file:
            json.dump(json_data, file, indent=4)
