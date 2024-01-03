import fasttext
import os
import json

file = "ft_train"


def prepare_train_file():
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    file_name = os.path.join(script_dir, 'data\\{}.txt'.format(file))
    if os.path.exists(file_name):
        return
    json_file = os.path.join(script_dir, 'data\\train_filtered.json')
    if not os.path.exists(json_file):
        print("no source file: {}".format(json_file))
        return
    with open(json_file, 'r', encoding='utf-8') as fin:
        loaded_obj = json.load(fin)
    if not os.path.exists(file_name):
        with open(file_name, "a", encoding='utf-8') as fout:
            for it in loaded_obj:
                text = it['text']
                for obj in it['labels']:
                    text += " __label__" + obj.replace(" ", "_")
                if not it['labels']:
                    text += " __label__"
                text += "\n"
                fout.write(text)


def ft_train():
    prepare_train_file()
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    file_name = os.path.join(script_dir, 'data\\{}.txt'.format(file))
    json_file = os.path.join(script_dir, 'data\\ft_validation.json.txt')

    model = fasttext.train_supervised(file_name, lr=1.0, epoch=75, loss='ova', wordNgrams=2, dim=200, thread=2, verbose=100)
    validation_file = os.path.join(script_dir, 'data\\validation_filtered.json')
    if not os.path.exists(validation_file):
        print("validation_filtered.json file does not exist")
        return
    with open(validation_file, 'r', encoding='utf-8') as fin:
        loaded_obj = json.load(fin)

    result_arr = []
    for it in loaded_obj:  # for meme in memes
        result = {}
        clusters = 1
        for i in range(1, 10):
            prediction = model.predict(it['text'], k=i)
            if prediction[1][i-1] >= 0.9:
                clusters = i
            else:
                break
        prediction = model.predict(it['text'], k=clusters)
        result['id'] = it['id']
        labels = []
        for label in prediction[0]:
            res_label = label.replace("__label__", "").replace("_", " ")
            if res_label:
                labels.append(res_label)
        result['labels'] = labels
        result_arr.append(result)
    with open(json_file, 'w', encoding='utf-8') as fout:
        json.dump(result_arr, fout)
