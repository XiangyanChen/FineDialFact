import random
import numpy as np
import torch
import os
import json
import pandas as pd

def write_output(output_pah, data, mode="a"):
    with open(output_pah, encoding="utf-8", mode=mode) as f:
        f.writelines(data)

def read_file(input_path):
    with open(input_path, encoding="utf-8", mode="r") as f:
        return f.readlines()

def read_json_file(input_path):
    with open(input_path, encoding="utf-8", mode="r") as f:
        text = f.read()
        lines = json.loads(text)
    return lines


def json_file_to_excel(input_path, output_path):
    data = read_json_file(input_path)

    df = pd.DataFrame(data)
    excel_path = output_path
    df.to_excel(excel_path, index=False)

    print(f"successful {excel_path}")

def data_to_excel(data, output_path, is_random=False, total_number=None):
    if (is_random == True):
        random.shuffle(data)
    if (total_number is not None):
        data = data[:total_number]

    df = pd.DataFrame(data)
    excel_path = output_path
    df.to_excel(excel_path, index=False)

    print(f"successful {excel_path}")

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def remove_file(path):
    if (os.path.exists(path)):
        os.remove(path)


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dictionary to store the results
    dict_of_lists = {}

    # Loop over each dictionary in the list
    for d in list_of_dicts:
        # Loop over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the result dictionary, initialize it with an empty list
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the list corresponding to the key
            dict_of_lists[key].append(value)

    return dict_of_lists
