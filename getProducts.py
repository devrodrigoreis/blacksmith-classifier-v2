import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC9hcGlkYXRhaW4tcHJvZC5mMXdzLmNvbS5iclwvYXBpXC9sb2dpbiIsImlhdCI6MTcxNDE4NzkzOSwibmJmIjoxNzE0MTg3OTM5LCJqdGkiOiJpZnBlTm9kc2w1QWFtWERsIiwic3ViIjoyODMsInBydiI6Ijg3ZTBhZjFlZjlmZDE1ODEyZmRlYzk3MTUzYTE0ZTBiMDQ3NTQ2YWEifQ.O7nW_pBeuOC3XcF8rWGEpqgcQAyTkah8QAHGp-OiGsY"
#os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in environment variables.")
    exit(1)

# API endpoint
URL = "http://apidatain-prod.f1ws.com.br/api/products"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": API_KEY
}

# Output file
output_file = "products_without_category.csv"

# Function to fetch products and process them
def fetch_and_process_products():
    current_page = 13
    last_page = None
    total_processed = 0
    total_matched = 0
    
    # Create/overwrite output file with header
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("product_id,product_name\n")
    
    print("Starting to fetch products...")
    
    while True:
        try:
            # Use an empty data payload or minimal required parameters
            payload = {}  # Empty payload or add minimal requirements
            
            # Make API request - separate page parameter for URL query and payload for body
            response = requests.get(f"{URL}?page={current_page}", headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Set last_page on first iteration
            if last_page is None:
                last_page = data.get("last_page", 1)
                print(f"Total pages to process: {last_page}")
            
            # Process products on this page
            products = data.get("data", [])
            
            if not products:
                print(f"No products found on page {current_page}. Ending process.")
                break
            
            matched_products = []
            
            for product in products:
                total_processed += 1
                
                categories = product.get("categories", [])
                
                # Check if no categories or only category with code 99
                if not categories or (len(categories) == 1 and categories[0].get("code") == "99"):
                    product_id = product.get("code", "")
                    product_name = product.get("name", "")
                    if product_name is None:
                        product_name = "No name provided"
                    else:
                        # Clean the product name for CSV format
                        product_name = str(product_name).replace(",", " ").replace("\n", " ")
                    
                    matched_products.append(f"{product_id},{product_name}")
                    total_matched += 1
            
            # Save matched products to file
            if matched_products:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(matched_products) + "\n")
            
            print(f"Processed page {current_page}/{last_page} - Found {len(matched_products)} matching products on this page")
            
            # Check if we've reached the last page
            if current_page >= last_page:
                break
            
            # Move to next page
            current_page += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {current_page}: {e}")
            # Retry after a longer delay
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    print(f"Processing complete. Processed {total_processed} products, found {total_matched} matching products.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    fetch_and_process_products()