import os
import csv
import json
import random
import string
import shutil

OUTPUT_DIR = 'data_fake'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_product_name():
    adjectives = ['Super', 'Ultra', 'Mega', 'Eco', 'Pro', 'Smart', 'Flex']
    nouns = ['Drill', 'Hammer', 'Saw', 'Paint', 'Glue', 'Tape', 'Ladder', 'Pump']
    return f"{random.choice(adjectives)} {random.choice(nouns)} {random.randint(100, 9000)}"

def random_category():
    l1 = ['Tools', 'Construction', 'Gardening', 'Electrical', 'Plumbing']
    l2 = ['Manual', 'Power', 'Accessories', 'Safety', 'Fixtures']
    l3 = ['Basic', 'Advanced', 'Professional', 'Home', 'Industrial']
    return f"{random.choice(l1)}>>{random.choice(l2)}>>{random.choice(l3)}"

def generate_csv(filename, headers, rows):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_json(filename, data):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    # 1. categories.csv
    # category_name,category_id
    categories = []
    category_ids = []
    for i in range(20):
        cat = random_category()
        cid = f"{random.randint(1, 10)}.{random.randint(1, 10)}.{random.randint(1, 10)}"
        categories.append([cat, cid])
        category_ids.append(cid)
    generate_csv(os.path.join(OUTPUT_DIR, 'categories.csv'), ['category_name', 'category_id'], categories)

    # 2. products.csv (and similar structure)
    # pid,product_name,product_category_tree
    products_data = []
    for i in range(50):
        pid = random.randint(1000, 99999)
        pname = random_product_name()
        pct = random.choice(categories)[0]
        products_data.append([pid, pname, pct])
    
    generate_csv(os.path.join(OUTPUT_DIR, 'products.csv'), ['pid', 'product_name', 'product_category_tree'], products_data)
    generate_csv(os.path.join(OUTPUT_DIR, 'products_train_unique.csv'), ['pid', 'product_name', 'product_category_tree'], products_data[:40])
    generate_csv(os.path.join(OUTPUT_DIR, 'products_eval_unique.csv'), ['pid', 'product_name', 'product_category_tree'], products_data[40:])

    # 3. CategorizedProducts.csv
    # product_id,product_name,category_name,category_id,confidence,model
    cat_products = []
    for i in range(20):
        pid = random.randint(1000, 99999)
        pname = random_product_name()
        idx = random.randint(0, len(categories)-1)
        cname = categories[idx][0]
        cid = categories[idx][1]
        conf = random.uniform(0.8, 0.99)
        model = 'fallback'
        cat_products.append([pid, pname, cname, cid, conf, model])
    generate_csv(os.path.join(OUTPUT_DIR, 'CategorizedProducts.csv'), ['product_id', 'product_name', 'category_name', 'category_id', 'confidence', 'model'], cat_products)

    # 4. UncatProducts.csv
    # product_title
    uncat = [[random_product_name()] for _ in range(10)]
    generate_csv(os.path.join(OUTPUT_DIR, 'UncatProducts.csv'), ['product_title'], uncat)

    # 5. uncategorized_products.csv
    # product_id,product_name
    uncat_prods = []
    for i in range(10):
        pid = random.randint(1000, 99999)
        pname = random_product_name()
        uncat_prods.append([pid, pname])
    generate_csv(os.path.join(OUTPUT_DIR, 'uncategorized_products.csv'), ['product_id', 'product_name'], uncat_prods)

    # 6. products_with_descriptions.csv
    # pid,product_name,product_category_tree,details
    desc_products = []
    for row in products_data[:10]:
        desc_products.append(row + [f"Description for {row[1]}"])
    generate_csv(os.path.join(OUTPUT_DIR, 'products_with_descriptions.csv'), ['pid', 'product_name', 'product_category_tree', 'details'], desc_products)

    # 7. return.json
    # { "current_page": 1, "data": [ { "id": 79941, "code": "052814", ... } ] }
    json_data = {
        "current_page": 1,
        "data": [
            {"id": random.randint(10000, 99999), "code": f"{random.randint(10000, 99999):06d}"}
            for _ in range(5)
        ]
    }
    generate_json(os.path.join(OUTPUT_DIR, 'return.json'), json_data)
    
    # 8. old_ds folder
    ensure_dir(os.path.join(OUTPUT_DIR, 'old_ds'))
    
    print(f"Fake dataset generated in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
