import stanza
import requests

nlp = stanza.Pipeline(lang='ru')


def ifpos(word, sent, close1, close2):

    neighbours = ''
    neighbour1 = sent.words[word.head+close1]
    neighbour2 = sent.words[word.head+close2]

    if neighbour1.pos in ('ADJ', 'NOUN', 'VERB'):
        neighbours1 = neighbour1.text

    if neighbour2.pos in ('ADJ', 'NOUN', 'VERB'):
        neighbours2 = neighbour2.text


    return neighbours1, neighbours2



def find_relatives(text, ent):
    ent_with_relatives = []

    for sent in nlp(text).sentences:
        for word in sent.words:
            if word.text in ent:
                try:
                    pos = word.pos

                    if len(sent.words) > 8:

                        if word.head == 0:
                            relative_words = ifpos(word, sent, close1=-1, close2=-2)
                            new_string = f'{relative_words[0]} {relative_words[1]} {word.text}'


                        elif word.head == 1:
                            relative_words = ifpos(word, sent, close1=-1, close2=1)
                            new_string = f'{relative_words[0]} {word.text} {relative_words[1]}'

                        else:
                            relative_words = ifpos(word, sent, close1=1, close2=2)
                            new_string = f'{word.text} {relative_words[0]} {relative_words[1]}'

                    else:
                        relative_words = 'Bad Application'
                        new_string = sent.text


                except:
                    relative_words = 'Bad Application'
                    new_string = sent.text


                result = (word.text, word.deprel, word.head, pos, relative_words, new_string)
                ent_with_relatives.append(result)
    
            else:
                pass
            
    if len(ent_with_relatives) > 1:
        result = ent_with_relatives[0][-1]
    elif len(ent_with_relatives) == 0 or ent_with_relatives == None:
        result = text
    else:
        result = ent_with_relatives[-1]
        
    return result

        
        
def implement(text, ents):

    output = find_relatives(text, ents)

    return output
