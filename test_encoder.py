import joblib
encoder = joblib.load("category_encoder.joblib")
print(len(encoder.classes_))  # Includes 'Unknown'


encoder = joblib.load("models/category_encoder.joblib")
print(len(encoder.classes_))  # Includes 'Unknown'


encoder = joblib.load("fallback_label_encoder.joblib")
print(len(encoder.classes_))  # Includes 'Unknown'