import pandas as pd
import re
from uuid import uuid4

# Load the dataset in chunks
def load_dataset(file_path, chunk_size=10000):
    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip')
        print(f"Loading dataset from {file_path} in chunks of {chunk_size} rows.")
        return chunks
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Fix inconsistencies in a chunk where possible (excluding misclassifications)
def fix_inconsistencies(df):
    df = df.copy()  # Avoid modifying the original chunk
    fixes_applied = []

    for index, row in df.iterrows():
        pid = row['pid']
        original_name = row['product_name']
        original_category = row['product_category_tree']

        # Fix 1: Remove trailing spaces
        if original_name.strip() != original_name:
            df.at[index, 'product_name'] = original_name.strip()
            fixes_applied.append([pid, original_name, original_category, "Removed trailing spaces from product name"])
        if original_category.strip() != original_category:
            df.at[index, 'product_category_tree'] = original_category.strip()
            fixes_applied.append([pid, original_name, original_category, "Removed trailing spaces from category"])

        # Fix 2: Convert all-uppercase product names to title case
        if original_name.isupper():
            df.at[index, 'product_name'] = original_name.title()
            fixes_applied.append([pid, original_name, original_category, "Converted product name to title case"])

        # Fix 3: Replace ampersands in category with 'e'
        if '&' in original_category:
            new_category = original_category.replace('&', 'e')
            df.at[index, 'product_category_tree'] = new_category
            fixes_applied.append([pid, original_name, original_category, "Replaced ampersand with 'e' in category"])

        # Fix 4: Correct typos
        if 'cemicolor' in original_name.lower():
            new_name = re.sub(r'cemicolor', 'Chemicolor', original_name, flags=re.IGNORECASE)
            df.at[index, 'product_name'] = new_name
            fixes_applied.append([pid, original_name, original_category, "Corrected typo: CEMICOLOR to CHEMICOLOR"])
        if 'col5una' in original_name.lower():
            new_name = re.sub(r'col5una', 'Coluna', original_name, flags=re.IGNORECASE)
            df.at[index, 'product_name'] = new_name
            fixes_applied.append([pid, original_name, original_category, "Corrected typo: COL5UNA to Coluna"])

    return df, pd.DataFrame(fixes_applied, columns=['pid', 'product_name', 'category', 'fix_applied'])

# Check for inconsistencies in a chunk (including tagging misclassifications)
def check_inconsistencies(df):
    inconsistencies = []

    def add_inconsistency(pid, product_name, category, issue):
        inconsistencies.append({
            'pid': pid,
            'product_name': product_name,
            'category': category,
            'issue': issue
        })

    # 1. Duplicate PIDs
    duplicate_pids = df[df['pid'].duplicated(keep=False)]
    for _, row in duplicate_pids.iterrows():
        add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Duplicate PID")

    # 2. Duplicate product names
    duplicate_names = df[df['product_name'].duplicated(keep=False)]
    for _, row in duplicate_names.iterrows():
        add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Duplicate product name")

    # 3. Formatting issues
    for _, row in df.iterrows():
        # Inconsistent category separator or depth
        category_levels = row['product_category_tree'].split('>>')
        if len(category_levels) != 3:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], f"Inconsistent category depth: {len(category_levels)} levels")
        if not re.match(r'^[^>]+>>[^>]+>>[^>]+$', row['product_category_tree']):
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Invalid category separator format")

        # Tag remaining uppercase names (in case not all were fixed)
        if row['product_name'].isupper():
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Product name all uppercase; potential formatting issue")

    # 4. Tag misclassifications (not fixed)
    for _, row in df.iterrows():
        product_name = row['product_name'].lower()
        category = row['product_category_tree'].lower()

        # Hinges in water tanks
        if 'dobradiça' in product_name and 'caixas d\'água' in category:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Misclassification: Hinges under water tanks")

        # Shoe grease in automotive
        if 'graxa calçados' in product_name and 'automotivo' in category:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Misclassification: Shoe grease in automotive")

        # Spray faucet in household faucets
        if 'torneira p/haste pulverização' in product_name and 'torneiras maquina de lavar e tanque' in category:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Misclassification: Spray faucet in household faucets")

    return pd.DataFrame(inconsistencies)

# Save inconsistencies and fixes to CSV
def save_output(df, output_file, output_type='inconsistencies'):
    if not df.empty:
        df.to_csv(output_file, index=False)
        print(f"{output_type.capitalize()} saved to {output_file}")
    else:
        print(f"No {output_type.lower()} found")

# Main function
def main():
    file_path = 'data/products.csv'  # Update with your actual path
    output_inconsistencies = f'inconsistencies_{uuid4()}.csv'
    output_fixes = f'fixes_{uuid4()}.csv'
    output_cleaned = 'cleaned_products.csv'
    chunk_size = 10000

    chunks = load_dataset(file_path, chunk_size)
    if chunks is None:
        return

    # Process chunks
    all_inconsistencies = []
    all_fixes = []
    cleaned_chunks = []
    pid_set = set()

    for chunk in chunks:
        # Check for duplicate PIDs across chunks
        for pid in chunk['pid']:
            if pid in pid_set:
                chunk.loc[chunk['pid'] == pid, 'issue'] = "Duplicate PID across chunks"
                all_inconsistencies.append(chunk[chunk['pid'] == pid][['pid', 'product_name', 'product_category_tree', 'issue']])
            pid_set.add(pid)

        # Fix inconsistencies (excluding misclassifications)
        cleaned_chunk, fixes = fix_inconsistencies(chunk)
        if not fixes.empty:
            all_fixes.append(fixes)

        # Check remaining inconsistencies (including tagged misclassifications)
        inconsistencies = check_inconsistencies(cleaned_chunk)
        if not inconsistencies.empty:
            all_inconsistencies.append(inconsistencies)

        cleaned_chunks.append(cleaned_chunk)

    # Combine and save results
    if all_fixes:
        final_fixes = pd.concat(all_fixes, ignore_index=True)
        save_output(final_fixes, output_fixes, 'fixes')
    if all_inconsistencies:
        final_inconsistencies = pd.concat(all_inconsistencies, ignore_index=True)
        save_output(final_inconsistencies, output_inconsistencies, 'inconsistencies')
    if cleaned_chunks:
        cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        save_output(cleaned_df, output_cleaned, 'cleaned dataset')

if __name__ == '__main__':
    main()