import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)              # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)                   # Remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)                 # Remove numbers/punctuation
    text = text.lower().strip()                             # Normalize case
    return text
# Expected CSV format: text,label (0=human, 1=AI-generated)
data = pd.read_csv("social_media_posts.csv")
data["text"] = data["text"].apply(preprocess_text)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, train_labels)
rf_preds = rf_model.predict(X_test_tfidf)
print("=== Random Forest Results ===")
print("Accuracy:", accuracy_score(test_labels, rf_preds))
print(classification_report(test_labels, rf_preds, target_names=["Human", "AI-Generated"]))
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
def train_transformer(model_name, tokenizer_class, model_class):
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=2)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    args = TrainingArguments(
        output_dir=f"./{model_name}_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir=f"./{model_name}_logs"
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    print(f"=== {model_name} Results ===")
    print("Accuracy:", accuracy_score(test_labels, pred_labels))
    print(classification_report(test_labels, pred_labels, target_names=["Human", "AI-Generated"]))
    model.save_pretrained(f"./{model_name}_detector")
    tokenizer.save_pretrained(f"./{model_name}_detector")
# Train BERT-based model
train_transformer("bert-base-uncased", BertTokenizer, BertForSequenceClassification)
# Train RoBERTa-based model
train_transformer("roberta-base", RobertaTokenizer, RobertaForSequenceClassification)
print("Model training and evaluation complete!")
