import os
import json
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'Exaggeration/Minimisation': 10,
    'Doubt': 11,
    'Appeal to fear/prejudice': 12,
    'Flag-waving': 13,
    'Reductio ad Hitlerum': 14,
    'Red herring': 15,
    'Presenting Irrelevant Data (Red Herring)': 15,
    'Bandwagon': 16,
    'Obfuscation, intentional vagueness, confusion': 17,
    'Straw man': 18,
    'Obfuscation, Intentional vagueness, Confusion': 17,
    "Misrepresentation of Someone's Position (Straw Man)": 18,
    'Slogans': 19,
}


class CustomDataset:
    def __init__(self, json_file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        script_dir = os.path.dirname(__file__)
        json_file = os.path.join(script_dir,json_file_path)

        with open(json_file, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']

        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_length).to(device)

        label_tensor = torch.zeros(len(label_mapping)).to(device)

        for i, label in enumerate(labels):
            label_index = label_mapping.get(label, -1)
            if label_index != -1:
                label_tensor[label_index] = 1


        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': label_tensor
        }



def train():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                num_labels=len(label_mapping))

    model = model.to(device)

    #Data load
    json_file_path = 'data/train_filtered.json'
    max_length = 128
    custom_dataset = CustomDataset(json_file_path, tokenizer, max_length)

    batch_size = 32
    dataloader = DataLoader(list(custom_dataset), batch_size=batch_size, shuffle=False)

    model.to(device)

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(dataloader) * 3  # Number of batches * number of epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    #train loop
    num_epochs = 10  # Set as needed
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
            batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

        script_dir = os.path.dirname(__file__)
        models_path = os.path.join(script_dir, 'models')

        model.save_pretrained(models_path)
        tokenizer.save_pretrained(models_path)


def validation():
    script_dir = os.path.dirname(__file__)

    model_path = os.path.join(script_dir, 'models')
    tokenizer_path = os.path.join(script_dir, 'models')

    json_file_path = 'data/validation_filtered.json'
    max_length = 128
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(label_mapping))
    model.to(device)

    val_dataset = CustomDataset(json_file_path, tokenizer, max_length)

    val_batch_size = 32  # Set as needed
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)

            batch_predictions = predictions.cpu().numpy().tolist()
            all_predictions.extend(batch_predictions)

    # Map predictions to labels based on threshold
    threshold = 0.5  # Adjust as needed
    binary_predictions = [[1 if score > threshold else 0 for score in example] for example in all_predictions]

    # Create a list of dictionaries with id and predicted labels
    id_list = [entry['id'] for entry in val_dataset.data]
    text_list = [entry['text'] for entry in val_dataset.data]

    label_keys = list(label_mapping.keys())
    result_list = [{'id': id, 'text': text_list[i],
                    'predicted_labels': [label_keys[j] for j in range(len(label_keys)) if
                                         binary_predictions[i][j] == 1]}
                   for i, id in enumerate(id_list)]

    result_list_scorer = [{'id': id,
                    'labels': [label_keys[j] for j in range(len(label_keys)) if
                                         binary_predictions[i][j] == 1]}
                   for i, id in enumerate(id_list)]

    #debug
    print("Length of label keys:", len(label_keys))
    print("Length of id list:", len(id_list))

    # Export predictions as JSON
    output_json_path = os.path.join(script_dir, 'data/predictions_distilbert.json')
    output_json_path_scorer = os.path.join(script_dir, 'data/predictions_distilbert_scorer.json')
    with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
        json.dump(result_list, output_json_file, ensure_ascii=False, indent=4)

    with open(output_json_path_scorer, 'w', encoding='utf-8') as output_json_file_scorer:
        json.dump(result_list_scorer, output_json_file_scorer, ensure_ascii=False, indent=4)


    output_txt_path = os.path.join(script_dir, 'data/predictions_distilbert.json.txt')

def extra_train_pretrained():
    script_dir = os.path.dirname(__file__)

    model_path = os.path.join(script_dir, 'models')
    tokenizer_path = os.path.join(script_dir, 'models')

    #pretrained
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(label_mapping))
    model.to(device)

    # Data load
    json_file_path = 'data/train_filtered.json'
    max_length = 128
    custom_dataset = CustomDataset(json_file_path, tokenizer, max_length)

    batch_size = 32
    dataloader = DataLoader(list(custom_dataset), batch_size=batch_size, shuffle=False)

    model.to(device)

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(dataloader) * 3  # Number of batches * number of epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # train loop
    num_epochs = 10  # Set as needed
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
                batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

        script_dir = os.path.dirname(__file__)
        models_path = os.path.join(script_dir, 'models')

        model.save_pretrained(models_path)
        tokenizer.save_pretrained(models_path)

def convert_json(json_file_path,end_filepath):
    script_dir = os.path.dirname(__file__)

    json_file = os.path.join(script_dir,json_file_path)

    with open(json_file, 'r',encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Convert the JSON data to a string
    text_data = json.dumps(data, indent=4)  # Adjust the indent parameter as needed

    end_file=os.path.join(script_dir,end_filepath)
    
    # Write the string to the output file
    with open(end_file, 'x',encoding='utf-8') as txt_file:
        txt_file.write(text_data)

    print("Conversion completed.")

if __name__ == '__main__':
    #extra_train_pretrained()
    #validation()
    convert_json('data/predictions_distilbert_scorer.json','data/predictions_distilbert_scorer.json.txt')

