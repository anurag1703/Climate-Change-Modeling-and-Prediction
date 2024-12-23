import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import joblib
import nltk


data_path = '/mnt/data/climate_nasa.csv'
data = pd.read_csv(data_path)

data = data.dropna(subset=['text', 'commentsCount'])  


def generate_sentiment_label(count):
    if count < 0:
        return 0  # Negative sentiment
    elif count == 0:
        return 1  # Neutral sentiment
    else:
        return 2  # Positive sentiment


data['sentiment'] = data['commentsCount'].apply(generate_sentiment_label)


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text


data['cleaned_text'] = data['text'].apply(preprocess_text)


X = data['cleaned_text']
y = data['sentiment']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_tokenized = [word_tokenize(text) for text in X_train]
X_test_tokenized = [word_tokenize(text) for text in X_test]


word2vec_model = Word2Vec(sentences=X_train_tokenized, vector_size=300, window=5, min_count=1, workers=4)
print("Word2Vec model trained.")


def generate_word2vec_embedding(tokens, word2vec_model):
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Return zero vector if no word matches


X_train_embeddings = np.array([generate_word2vec_embedding(tokens, word2vec_model) for tokens in X_train_tokenized])
X_test_embeddings = np.array([generate_word2vec_embedding(tokens, word2vec_model) for tokens in X_test_tokenized])


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        _, (hidden, _) = self.rnn(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  
        output = self.fc(hidden_cat)
        return self.softmax(output)


class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


train_dataset = SentimentDataset(X_train_embeddings, y_train)
test_dataset = SentimentDataset(X_test_embeddings, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


input_dim = word2vec_model.vector_size
hidden_dim = 128
output_dim = 3  

model = RNNClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = model(embeddings)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


model_dir = r'C:\Users\anura\Desktop\Project 4- Climate Change Modeling\models'
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, 'rnn_multiclass_sentiment_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"RNN Multiclass Model saved at {model_save_path}")
