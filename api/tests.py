import joblib

le = joblib.load('e:/Blacksmith/blacksmith-classifier/category_encoder.joblib')
print(len(le.classes_))
print(le.classes_[:5])  # First 5 classes

import pandas as pd
df = pd.read_csv('e:/Blacksmith/blacksmith-classifier/data/categories.csv')
print(df['category_name'].nunique())
print(df['category_name'].head())

