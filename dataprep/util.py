import os
import re

def create_pattern_dict(FILE_PATH):
    """
    Create a dictionary from a list of patterns.

    Args:
        patterns (list): A list of patterns.

    Returns:
        dict: A dictionary with patterns as keys and True as values.
    """

    print(f"Looking for file at:\n{FILE_PATH}\n")

    # === STEP 1: Prepare regex pattern for transaction lines ===
    pattern = re.compile(
        r"^(\S+\s+\S+),"              # Timestamp
        r"(\S+),"                     # From Bank
        r"(\S+),"                     # From Account
        r"(\S+),"                     # To Bank
        r"(\S+),"                     # To Account
        r"([\d.]+),"                  # Amount Received
        r"([^,]+),"                   # Receiving Currency
        r"([\d.]+),"                  # Amount Paid
        r"([^,]+),"                   # Payment Currency
        r"([^,]+),"                   # Payment Format
        r"(\d+)$"                     # Is Laundering
    )

    # === STEP 2: Parse the file into dictionary ===
    pattern_dict = {}
    current_key = None
    current_values = []
    all_keys = {}

    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Detect beginning of a new laundering attempt section
                if line.startswith("BEGIN LAUNDERING ATTEMPT -"):
                    current_key = line.split("-", 1)[1].strip()
                    if current_key not in all_keys.keys():
                        all_keys[current_key] = 1
                    else:   
                        all_keys[current_key] += 1
                        current_key = f"{current_key}_dup{all_keys[current_key]}"
                        print(f"Warning: Duplicate key found: {current_key}")
                    
                    current_values = []
                    continue

                # Detect end of section
                if line.startswith("END LAUNDERING ATTEMPT -"):
                    if current_key is not None:
                        # Save all collected transactions at the end
                        pattern_dict[current_key] = current_values
                    current_key = None
                    current_values = []
                    continue

                # Process transaction line (only inside a section)
                if current_key:
                    m = pattern.match(line)
                    if m:
                        current_values.append(m.groups())
                    else:
                        print(f"Line did NOT match regex: {line}")

    else:
        print("File not found!")

    return pattern_dict

def show_pattern_dict(pattern_dict):
    """
    Display the contents of a pattern dictionary.

    Args:
        pattern_dict (dict): A dictionary with patterns as keys.
    """
    for key, vals in pattern_dict.items():
        print(f"\n=== {key} ===")
        print(f"Transactions: {len(vals)}")
        for row in vals:  # show only first 2 per section
            print(row)


def filter_dict(d: dict, substring: str):
    list_in_list =[value for key, value in d.items() if substring in str(key)]
    return [item for sublist in list_in_list for item in sublist]

