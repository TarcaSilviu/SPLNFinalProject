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

    with open(file_name, "a", encoding='utf-8') as fout:
        for it in loaded_obj:
            text = it['text']
            for obj in it['labels']:
                text += " __label__" + obj.replace(" ", "_")
            text += "\n"
            fout.write(text)


def ft_train():
    prepare_train_file()
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    file_name = os.path.join(script_dir, 'data\\{}.txt'.format(file))
    print(file_name)
    model = fasttext.train_supervised(file_name)
    #print(model.words)
    print()
    print(model.labels)
