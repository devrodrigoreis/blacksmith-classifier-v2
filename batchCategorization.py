import csv
import requests
import pandas as pd
import time
from tqdm import tqdm

# Configuration
API_URL = "http://localhost:8000/predict"  # Adjust if your service runs on a different host/port
INPUT_CSV = "data/uncategorized_products.csv"
OUTPUT_CSV = "data/CategorizedProducts.csv"
BATCH_SIZE = 100  # Number of records to process at once before saving
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def categorize_product(product_name):
    """Send a product to the API for categorization."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(API_URL, json={"product_name": product_name}, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error processing '{product_name}': {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to categorize after {MAX_RETRIES} attempts")
                return {
                    "bert_prediction": {"category_name": "Error", "category_id": "N/A", "confidence": 0.0},
                    "fallback_prediction": {"category_name": "Error", "category_id": "N/A", "confidence": 0.0},
                    "recommended_prediction": {"category_name": "Error", "category_id": "N/A", "confidence": 0.0, "recommended": True}
                }

def main():
    try:
        # Read input CSV
        print(f"Reading products from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        products = df[['product_id', 'product_name']].values.tolist()  # Get both columns
        
        results = []
        
        # Process products
        print(f"Processing {len(products)} products...")
        for i, (product_id, product_name) in enumerate(tqdm(products)):
            response = categorize_product(product_name)
            
            # Extract the recommended prediction
            rec_pred = response['recommended_prediction']
            
            # Store results
            results.append({
                'product_id': product_id,
                'product_name': product_name,
                'category_name': rec_pred['category_name'],
                'category_id': rec_pred['category_id'],
                'confidence': rec_pred['confidence'],
                'model': rec_pred.get('model', 'unknown')
            })
            
            # Save intermediate results in batches
            if (i + 1) % BATCH_SIZE == 0 or i == len(products) - 1:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
                print(f"Saved progress: {i+1}/{len(products)} products")
        
        print(f"Categorization completed. Results saved to {OUTPUT_CSV}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'results' in locals() and results:
            # Save whatever we've processed so far
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"Partial results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()