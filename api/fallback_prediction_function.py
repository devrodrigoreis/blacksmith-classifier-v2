from sklearn.preprocessing import LabelEncoder
import joblib
import torch
import nltk
import re
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('portuguese'))

# Preprocess text function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

class TextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = torch.nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Load the saved model, label encoder, and vocabulary
model = torch.load('models/fallback_model.pth')
label_encoder = joblib.load('models/fallback_label_encoder.joblib')
vocab = joblib.load('models/fallback_vocab.joblib')

def predict_category(product_name):
    # Preprocess the input text
    preprocessed_text = preprocess_text(product_name)
    
    # Tokenize and convert to tensor
    tokens = nltk.word_tokenize(preprocessed_text)
    input_tensor = torch.tensor([[vocab.get(token, vocab['<unk>']) for token in tokens]], dtype=torch.long)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # Convert predicted index back to category name
    predicted_category = label_encoder.inverse_transform(predicted.numpy())
    
    return predicted_category[0]