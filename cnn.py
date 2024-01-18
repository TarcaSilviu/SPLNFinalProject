import os
import json
from sre_parse import Tokenizer
from fastai.layers import Embedding
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def prepare_train_file(file):
    script_dir = os.path.dirname(__file__)
    file_name = os.path.join(script_dir, f'data\\{file}.txt')
    if os.path.exists(file_name):
        return
    json_file = os.path.join(script_dir, 'data\\train_filtered.json')
    if not os.path.exists(json_file):
        print(f'No source file: {json_file}')
        return
    with open(json_file, 'r', encoding='utf-8') as fin:
        loaded_obj = json.load(fin)
    if not os.path.exists(file_name):
        with open(file_name, 'a', encoding='utf-8') as fout:
            for it in loaded_obj:
                text = it['text']
                for obj in it['labels']:
                    text += f" __label__{obj.replace(' ', '_')}"
                if not it['labels']:
                    text += ' __label__'
                text += '\n'
                fout.write(text)

def cnn_train(file, epochs=40):
    prepare_train_file(file)
    script_dir = os.path.dirname(__file__)
    file_name = os.path.join(script_dir, f'data\\{file}.txt')
    json_file = os.path.join(script_dir, f'data\\{file}_validation_cnn.json.txt')

    # Încărcarea datelor de antrenare
    with open(file_name, 'r', encoding='utf-8') as fin:
        train_data = fin.readlines()

    # Extrage etichetele și textul pentru antrenare
    examples = []
    for line in train_data:
        labels = line.strip().split(' ')[-1].split('__label__')[1:]
        text = ' '.join(line.strip().split(' ')[:-1])
        examples.append({'text': text, 'labels': labels})

    # Separă etichetele și textul
    texts = [example['text'] for example in examples]
    labels = [example['labels'] for example in examples]

    # Convertiți etichetele la o reprezentare binară pentru antrenare CNN
    mlb = MultiLabelBinarizer()
    labels_bin = mlb.fit_transform(labels)

    # Tokenizare și încorporare text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences)

    # Definirea modelului CNN
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=padded_sequences.shape[1]))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(mlb.classes_), activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Antrenarea modelului CNN
    model.fit(padded_sequences, labels_bin, epochs=epochs, validation_split=0.2)

    # Încărcarea datelor de validare
    validation_file = os.path.join(script_dir, 'data\\validation_filtered.json')
    if not os.path.exists(validation_file):
        print(f'{file}_validation.json file does not exist')
        return
    with open(validation_file, 'r', encoding='utf-8') as fin:
        loaded_obj = json.load(fin)

    result_arr = []
    for it in loaded_obj:
        result = {}
        text = it['text']

        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=padded_sequences.shape[1])

        # Make predictions with the CNN model
        prediction = model.predict(padded_sequence)[0]

        # Sort labels based on their predicted probabilities
        sorted_labels = np.array(mlb.classes_)[np.argsort(prediction)[::-1]]

        non_empty_labels = [label.replace('_', ' ') for label in sorted_labels[:3] if label]

        if non_empty_labels:
            result['id'] = it['id']
            result['labels'] = non_empty_labels  # Take the top 3 non-empty labels
            result_arr.append(result)

    # Save the results to a JSON file
    with open(json_file, 'w', encoding='utf-8') as fout:
        json.dump(result_arr, fout)