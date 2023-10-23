from sentiment import Sentiment_classification, Emotion_detection, Toxicity_detection
from dataset import VKPostsDataset
from preprocessing import clean, find_congratulation, emoji2text

#from deep_translator import GoogleTranslator

from tqdm import tqdm

import json
import pandas as pd
import requests
import re

import argparse

parser = argparse.ArgumentParser(description='Sentiment analysis & Emotion detection & Toxicity & Spam')
parser.add_argument('load_path', type=str, help='Path to the dataset for annotation (with file name)')
parser.add_argument('save_path', type=str, help='Path to save the result (with file name)')
parser.add_argument('--check_congrats', type=bool, default=False, help='Detect posts with congratulations')
parser.add_argument('--with_entities', type=bool, default=False, help='if you want to process a dataset with entities, type True')
args = parser.parse_args()

tqdm.pandas()


def fill_result(clf, text):
    try:
        res = clf.classify_text(text)
    except:
        res = None

    return res

file = args.load_path  # path to file
#r = requests.get(str(file))  # link to json file

with open(file, 'r') as f:
    #f = json.loads(f)
    df = pd.read_json(f)

df['text'] = df['text'].apply(clean)

if args.with_entities == True:
    with_entities = True
else:
    with_entities = False
dataset = VKPostsDataset(data=df, with_entities=with_entities)


clf = Sentiment_classification()
emclf = Emotion_detection()
toxclf = Toxicity_detection()


df['sentiment'] = df['text'].progress_apply(lambda text : fill_result(clf=clf, text=text))  ## Анализ тональности
df['emotion'] = df['text'].progress_apply(lambda text : fill_result(clf=emclf, text=text))  ## Выявление эмоций
df['toxicity'] = df['text'].progress_apply(lambda text : fill_result(clf=toxclf, text=text))  ## Ввыявление токсичности

if args.check_congrats == True:
    df['is_congratulation'] = df['text'].progress_apply(find_congratulation)  ## выявляет сообщения с поздравлениями


## Спам-фильтр пока отключен. Я его переделываю
## translator = GoogleTranslator(source='ru', target='en')
## df['spam'] = df['text'].progress_apply(check_spam)


df.to_json(args.save_path)


