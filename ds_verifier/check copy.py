
import pandas as pd
import re
from uuid import uuid4

# Load the dataset in chunks
def load_dataset(file_path, chunk_size=10000):
    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        return chunks
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Check for inconsistencies in a chunk
def check_inconsistencies(df):
    inconsistencies = []

    def add_inconsistency(pid, product_name, category, issue):
        inconsistencies.append({
            'pid': pid,
            'product_name': product_name,
            'category': category,
            'issue': issue
        })

    # 1. Duplicate PIDs (checked across chunks later)
    duplicate_pids = df[df['pid'].duplicated(keep=False)]
    for _, row in duplicate_pids.iterrows():
        add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Duplicate PID")

    # 2. Duplicate product names
    duplicate_names = df[df['product_name'].duplicated(keep=False)]
    for _, row in duplicate_names.iterrows():
        add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Duplicate product name")

    # 3. Formatting issues
    for _, row in df.iterrows():
        # Trailing spaces
        if row['product_name'].strip() != row['product_name']:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Trailing spaces in product name")
        if row['product_category_tree'].strip() != row['product_category_tree']:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Trailing spaces in category")

        # Ampersand in category
        if '&' in row['product_category_tree']:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Ampersand in category; inconsistent format")

        # Inconsistent category separator or depth
        category_levels = row['product_category_tree'].split('>>')
        if len(category_levels) != 3:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], f"Inconsistent category depth: {len(category_levels)} levels")
        if not re.match(r'^[^>]+>>[^>]+>>[^>]+$', row['product_category_tree']):
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Invalid category separator format")

        # Inconsistent case in product name (e.g., all uppercase or mixed)
        if row['product_name'].isupper():
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Product name all uppercase; potential formatting issue")

    # 4. Misclassifications
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

    # 5. Typos
    for _, row in df.iterrows():
        product_name = row['product_name'].lower()
        if 'col5una' in product_name:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Typo in name: COL5UNA likely COLUNA")
        if 'cemicolor' in product_name:
            add_inconsistency(row['pid'], row['product_name'], row['product_category_tree'], "Brand spelling inconsistency: CEMICOLOR vs CHEMICOLOR")

    return pd.DataFrame(inconsistencies)

# Save inconsistencies to CSV
def save_inconsistencies(inconsistencies_df, output_file):
    if not inconsistencies_df.empty:
        inconsistencies_df.to_csv(output_file, index=False)
        print(f"Inconsistencies saved to {output_file}")
    else:
        print("No inconsistencies found")

# Main function
def main():
    file_path = 'products.csv'  # Replace with your full dataset path
    output_file = f'inconsistencies_{uuid4()}.csv'  # Unique output file name
    chunk_size = 10000  # Adjust if needed

    chunks = load_dataset(file_path, chunk_size)
    if chunks is None:
        return

    # Process chunks and combine inconsistencies
    all_inconsistencies = []
    pid_set = set()  # Track PIDs across chunks for duplicate checking
    for chunk in chunks:
        # Check for duplicate PIDs across chunks
        for pid in chunk['pid']:
            if pid in pid_set:
                chunk.loc[chunk['pid'] == pid, 'issue'] = "Duplicate PID across chunks"
                all_inconsistencies.append(chunk[chunk['pid'] == pid][['pid', 'product_name', 'product_category_tree', 'issue']])
            pid_set.add(pid)

        # Check other inconsistencies
        inconsistencies = check_inconsistencies(chunk)
        if not inconsistencies.empty:
            all_inconsistencies.append(inconsistencies)

    # Combine and save results
    if all_inconsistencies:
        final_inconsistencies = pd.concat(all_inconsistencies, ignore_index=True)
        save_inconsistencies(final_inconsistencies, output_file)
    else:
        print("No inconsistencies found in any chunk")

if __name__ == '__main__':
    main()