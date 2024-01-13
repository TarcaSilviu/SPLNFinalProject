import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

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

def svm_train(file, C_value=1.0, gamma_value='scale', kernel_value='linear'):
    prepare_train_file(file)
    script_dir = os.path.dirname(__file__)
    file_name = os.path.join(script_dir, f'data\\{file}.txt')
    json_file = os.path.join(script_dir, f'data\\{file}_validation.json.txt')

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

    # Convertiți etichetele la un vector unidimensional
    labels = np.array(labels).ravel()

    # Antrenarea modelului SVM cu un pipeline TF-IDF
    model = make_pipeline(TfidfVectorizer(), SVC(probability=True, kernel=kernel_value, C=C_value, gamma=gamma_value))
    model.fit(texts, labels)

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

        # Realizarea unei predicții cu modelul SVM
        prediction = model.predict_proba([text])[0]

        # Se sortează etichetele în funcție de probabilitatea lor
        sorted_labels = model.classes_[prediction.argsort()[::-1]]

        non_empty_labels = [label.replace('_', ' ') for label in sorted_labels[:3] if label]

        if non_empty_labels:
            result['id'] = it['id']
            result['labels'] = non_empty_labels  # Ia primele 3 etichete ne-goale
            result_arr.append(result)

    # Salvarea rezultatelor într-un fișier JSON
    with open(json_file, 'w', encoding='utf-8') as fout:
        json.dump(result_arr, fout)