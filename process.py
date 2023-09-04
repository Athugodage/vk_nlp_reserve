from razdel import sentenize
from tqdm import tqdm
from sentiment import Sentiment_classification, Emotion_detection, Toxicity_detection

import json
import requests

#https://files.deeppavlov.ai/tmp/processed_comments.json
#https://files.deeppavlov.ai/tmp/processed_posts.json

r = requests.get(str(input()))  # link to json file
data = json.loads(r.text)

total_result = {}
    
    
clf = Sentiment_classification()
emclf = Emotion_detection()
toxclf = Toxicity_detection()

num = 0

for n in tqdm(range(len(data))):
    text = data[n]['text']
    sents = list(sentenize(str(text)))

    for entity in data[n]['entity_info']:
        for sent in sents:
            if sent.start < entity['offsets'][0] < entity['offsets'][1] < sent.stop:
                sentence = sent.text
                message = data[n]['text'] 
                entity_text = entity['substring']
                

                context = get_string.implement(sentence, entity)
                try:
                    result_clf = clf.classify_text(sent.text)
                except:
                    result_clf = None
                
                try:
                    result_emclf = emclf.classify_text(sent.text)
                except:
                    result_emclf = None
                    
                try:
                    result_toxclf = toxclf.classify_text(sent.text)
                except:
                    result_toxclf = None
                
                num += 1
                
                dictionary = {'phrase (context)': context,
                              'message': message,
                              'message_number' : n,
                              'entity': entity_text,
                              'entity_tag' : entity['tag'],
                              'sentiment' : result_clf,
                              'emotion' : result_emclf,
                              'toxicity' : result_toxclf}
                              
                total_result[num] = dictionary


with open(f'processed_posts.json', 'w') as f:
    json.dump(total_result, f)
                            
