import os
import json
import string, time
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords


def remove_punct(text):
    exclude = string.punctuation
    for char in exclude:
        text = text.replace(char, '')
    return text


def correct_text(incorrect_text):
    text = TextBlob(incorrect_text)
    text = text.correct()
    return text


def do_nltk(text):
    stop_words = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    words_without_stop_word = []
    for word in tokens:
        if word in stop_words:
            continue
        words_without_stop_word.append(word)
    return ' '.join(words_without_stop_word)


def process(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    file_name = os.path.join(script_dir, 'data\\{}_filtered.json'.format(file))
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as fin:
            loaded_obj = json.load(fin)
            for it in loaded_obj:
                print(it)
        return
    json_file = os.path.join(script_dir, 'data\\{}.json'.format(file))
    if not os.path.exists(json_file):
        print("no source file: {}".format(json_file))
        return
    with open(json_file, 'r', encoding='utf-8') as fin:
        loaded_obj = json.load(fin)
    result_arr = []
    for it in loaded_obj: # for meme in memes
        print(it)
        it['text'] = do_nltk(remove_punct(it['text'].lower().replace("\\n", " ").replace("'s", "")))
        if 'link' in it:
            del it['link']
        print(it)
        result_arr.append(it)
    with open(file_name, 'w', encoding='utf-8') as fout:
        loaded_obj = json.dump(result_arr, fout)


