import pandas as pd
import re
import csv

# Load the dataset
try:
    df = pd.read_csv('data/products.csv', sep=',', encoding='utf-8', on_bad_lines='skip')
    print(f"Loaded {len(df)} rows successfully.")
except FileNotFoundError:
    print("Error: 'products.csv' not found. Please ensure the file is in the working directory.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Initialize list to store inconsistencies
inconsistencies = []

# Helper functions for checks
def check_uppercase(text):
    return text.isupper()

def check_trailing_spaces(text):
    return text.endswith(' ')

def check_ampersand(text):
    return '&' in text

def check_category_depth(category):
    return len(category.split('>>')) != 3

def check_misclassification(row):
    name = row['product_name'].lower()
    category = row['product_category_tree'].lower()
    if 'dobradiça' in name and 'caixas d\'água' in category:
        return "Misclassification: Hinges under water tanks"
    if 'torneira spray' in name and 'torneiras para pia' in category:
        return "Misclassification: Spray faucet in household faucets"
    if 'graxa para sapato' in name and 'sapato' not in category:
        return "Misclassification: Shoe grease not in shoe category"
    return None

def check_typos(row):
    name = row['product_name'].lower()
    if 'cemicolor' in name:
        return "Brand spelling inconsistency: CEMICOLOR vs CHEMICOLOR"
    if 'col5una' in name:
        return "Potential typo: COL5UNA in product name"
    return None

# Perform checks
for index, row in df.iterrows():
    pid = row['pid']
    name = row['product_name']
    category = row['product_category_tree']

    # Check duplicates
    if df['pid'].duplicated().any():
        if df[df['pid'] == pid].index[0] != index:
            inconsistencies.append([pid, name, category, "Duplicate PID"])
    if df['product_name'].duplicated().any():
        if df[df['product_name'] == name].index[0] != index:
            inconsistencies.append([pid, name, category, "Duplicate product name"])

    # Check formatting issues
    if check_uppercase(name):
        inconsistencies.append([pid, name, category, "Product name all uppercase; potential formatting issue"])
    if check_trailing_spaces(name):
        inconsistencies.append([pid, name, category, "Trailing spaces in product name"])
    if check_ampersand(category):
        inconsistencies.append([pid, name, category, "Ampersand in category; inconsistent format"])
    if check_category_depth(category):
        inconsistencies.append([pid, name, category, "Category depth inconsistent (not 3 levels)"])

    # Check misclassifications
    misclassification = check_misclassification(row)
    if misclassification:
        inconsistencies.append([pid, name, category, misclassification])

    # Check typos
    typo = check_typos(row)
    if typo:
        inconsistencies.append([pid, name, category, typo])

# Save inconsistencies to CSV
with open('inconsistencies.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['pid', 'product_name', 'category', 'issue'])
    writer.writerows(inconsistencies)

print(f"Found {len(inconsistencies)} inconsistencies. Saved to 'inconsistencies.csv'.")