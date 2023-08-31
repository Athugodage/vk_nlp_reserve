from sentiment import Sentiment_classification, Emotion_detection, Toxicity_detection

def nlp_module(text):
    clf = Sentiment_classification()
    result_clf = clf.classify_text(text)

    emlcf = Emotion_detection()
    result_emclf = emlcf.classify_text(text)

    toxclf = Toxicity_detection()
    result_toxclf = toxclf.classify_text(text)


    # first element = sentiment analysis,
    # second = emotion detection
    # third = toxicity

    return [result_clf, result_emclf, result_toxclf]


if __name__ == "__main__":
    text = str(input())  # Пример: 'Актеру, игравшего Тони Сопрано удалось создать образ гангстера с человеческим лицом'
    result_clf, result_emclf, result_toxclf = nlp_module(text)

    print('Text: ', text)
    print('Sentiment analysis result: ', result_clf)
    print('Emotion detection result: ', result_emclf)
    print('Toxicity detection result: ', result_toxclf)