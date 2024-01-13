import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def load_bert_model(model_name, num_labels):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def preprocess_text(text, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs


def classify_labels(text, model, tokenizer):
    inputs = preprocess_text(text, tokenizer)
    outputs = model(**inputs)
    probabilities = sigmoid(outputs.logits)
    return probabilities


def calculate_precision(predictions, labels):
    threshold = 0.5
    binary_predictions = (predictions > threshold).float()

    TP = (binary_predictions * labels).sum(dim=0)
    FP = (binary_predictions * (1 - labels)).sum(dim=0)

    precision = TP / (TP + FP + 1e-10)  # Adding a small epsilon to avoid division by zero

    return precision


label_mapping = {
    'Loaded Language': 0,
    'Black-and-white Fallacy/Dictatorship': 1,
    'Glittering generalities (Virtue)': 2,
    'Thought-terminating cliché': 3,
    'Whataboutism': 4,
    'Causal Oversimplification': 5,
    'Smears': 6,
    'Name calling/Labeling': 7,
    'Appeal to authority': 8,
    'Repetition': 9,
    'Exaggeration or minimization': 10,
    'Doubt': 11,
    'Appeal to fear/prejudice': 12,
    'Flag-waving': 13,
    'Reductio ad Hitlerum': 14,
    'Red herring': 15,
    'Bandwagon': 16,
    'Obfuscation, intentional vagueness, confusion': 17,
    'Straw man': 18,
}


class CustomDataset:
    def __init__(self, json_file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item['text']
        labels = item['labels']

        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_length)
        label_tensor = torch.zeros(len(label_mapping))
        for i, label in enumerate(labels):
            label_index = label_mapping.get(label, -1)
            if label_index != -1:
                label_tensor[label_index] = 1

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': label_tensor
        }


def test():
    model_name = "bert-base-cased"
    num_labels = 19
    model, tokenizer = load_bert_model(model_name, num_labels)
    custom_dataset = CustomDataset('data/train_filtered.json', tokenizer, 128)
    val_dataset = CustomDataset('data/validation_filtered.json', tokenizer, 128)
   # train_dataset, val_dataset = train_test_split(custom_dataset)
    batch_size = 32
    train_dataloader = DataLoader(list(custom_dataset), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(list(val_dataset), batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(total_loss)

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                probabilities = sigmoid(logits)
                all_predictions.append(probabilities.cpu())
                all_labels.append(labels.cpu())
                val_loss += outputs.loss.item()
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        precision = calculate_precision(all_predictions, all_labels)
        average_val_loss = val_loss / len(val_dataloader)
        print(f"PRECISION IS {precision}")
        print(f"Epoch {epoch + 1}, Average Validation Loss: {average_val_loss}")
