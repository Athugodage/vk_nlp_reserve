from sentiment import Sentiment_classification, Emotion_detection, Toxicity_detection
from dataset import VKPostsDataset
from preprocessing import clean, find_congratulation, emoji2text

#from deep_translator import GoogleTranslator

from tqdm import tqdm

import json
import pandas as pd
import requests
import re


tqdm.pandas()


def fill_result(clf, text):
    try:
        res = clf.classify_text(text)
    except:
        res = None

    return res

file = str(input())  # path to file
r = requests.get(str(file))  # link to json file
f = json.loads(r.text)
df = pd.read_json(f)

df['text'] = df['text'].apply(clean)

dataset = VKPostsDataset(data=df, with_entities=True)


clf = Sentiment_classification()
emclf = Emotion_detection()
toxclf = Toxicity_detection()


df['sentiment'] = df['text'].progress_apply(lambda text : fill_result(clf=clf, text=text))  ## Анализ тональности
df['emotion'] = df['text'].progress_apply(lambda text : fill_result(clf=emclf, text=text))  ## Выявление эмоций
df['toxicity'] = df['text'].progress_apply(lambda text : fill_result(clf=toxclf, text=text))  ## Ввыявление токсичности

df['is_congratulation'] = df['text'].progress_apply(find_congratulation)  ## выявляет сообщения с поздравлениями


## Спам-фильтр пока отключен. Я его переделываю
## translator = GoogleTranslator(source='ru', target='en')
## df['spam'] = df['text'].progress_apply(check_spam)


df.to_json('output.json')


