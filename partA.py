from nltk.corpus import brown
from collections import Counter
from itertools import compress
import nltk

flatten = lambda t: [item for sublist in t for item in sublist]
news = list(brown.tagged_sents(categories='news'))
words = list(set(brown.words()))
tags = []
size = int(len(news)*0.9)
training = flatten(news[:size])
training_words = set([x[0] for x in training])
test = flatten(news[size:])
for word in words:
    #wordtagtuples has all the tuples from the training data where the first element in the tuple is word
    is_equal_to_word = [(x[0] == word) for x in training]
    #if word is an unknown word
    if not any(is_equal_to_word):
        tags.append('NN')
    else:
        wordtagtuples = list(compress(training, is_equal_to_word))
        highest_freq_tag = Counter([x[1] for x in wordtagtuples]).most_common(1)[0][0]
        tags.append(highest_freq_tag)

unigram_model = dict(zip(words, tags))
known_accuracy = 0
unknown_accuracy = 0
known_size = 0
unknown_size = 1
for tuple in test:
    if tuple[0] in training_words:
        known_size+=1
        if tuple[1] == unigram_model.get(tuple[0]):
            known_accuracy += 1
    else:
        unknown_size+=1
        if tuple[1] == unigram_model.get(tuple[0]):
            unknown_accuracy+=1
accuracy = (known_accuracy+unknown_accuracy)/(known_size+unknown_size)
known_accuracy /= known_size
unknown_accuracy /= unknown_size
print(unknown_accuracy)
print(known_accuracy)
print(accuracy)