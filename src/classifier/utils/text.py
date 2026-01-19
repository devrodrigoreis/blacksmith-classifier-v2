
import re
import nltk
from nltk.corpus import stopwords
import logging

# Ensure resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('portuguese'))

def preprocess_text(text, remove_stopwords=True):
    """
    Cleans and normalizes text.
    
    Args:
        text (str): Input text.
        remove_stopwords (bool): Whether to remove Portuguese stopwords.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars (keep alphanumeric and whitespace)
    # Using [^a-z0-9\s] to match trainer2 behavior. 
    # Fallback used [^\w\s] which includes underscores.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    if remove_stopwords:
        text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    
    return text
