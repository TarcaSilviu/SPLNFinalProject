import torch
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
import tensorflow as tf
import numpy as np
def load_bert_model(model_name, num_labels):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

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
    'Exaggeration/Minimisation': 10,
    'Doubt': 11,
    'Appeal to fear/prejudice': 12,
    'Flag-waving': 13,
    'Reductio ad Hitlerum': 14,
    'Presenting Irrelevant Data (Red Herring)': 15,
    'Bandwagon': 16,
    'Obfuscation, Intentional vagueness, Confusion': 17,
    "Misrepresentation of Someone's Position (Straw Man)": 18,
    'Slogans': 19,
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
            'ids': self.data[idx]['id'],
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': label_tensor
        }


def test():
    model_name = "bert-base-cased"
    num_labels = 20
    model, tokenizer = load_bert_model(model_name, num_labels)
    custom_dataset = CustomDataset('data/train_filtered.json', tokenizer, 128)
    val_dataset = CustomDataset('data/validation_filtered.json', tokenizer, 128)
   # train_dataset, val_dataset = train_test_split(custom_dataset)
    batch_size = 32
    train_dataloader = DataLoader(list(custom_dataset), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(list(val_dataset), batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    model.eval()
    val_loss = 0.0
    final_precision = 0
    iteration =0
    result_arr =[]
    with torch.no_grad():
        for batch in val_dataloader:
            iteration = iteration+1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            probs = tf.nn.softmax(logits.cpu(), axis=-1)
            batch_precision = 0

            for i in range(len(probs)):
                result = {}

                max_position = np.argmax(probs[i])
                result['id'] = batch['ids'][i]
                result['labels'] = []
                if labels[i][max_position] == 1:
                    label_key = next(key for key, value in label_mapping.items() if value == max_position)
                    result['labels'] = [label_key]
                    batch_precision = batch_precision+1
                result_arr.append(result)
            final_precision = final_precision + (batch_precision/32)
            val_loss += outputs.loss.item()

    script_dir = os.path.dirname(__file__)
    json_file = os.path.join(script_dir, 'data\\bert_validation.json.txt')
    with open(json_file, 'w', encoding='utf-8') as fout:
        json.dump(result_arr, fout)
    print('FINAL PRECISION')
    print(final_precision/iteration)
    average_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}, Average Validation Loss: {average_val_loss}")
