from nltk.corpus import brown
import numpy as np
from collections import Counter


# start by cleaning all '-' and '+' from the tags
sep1 = '-'
sep2 = '+'
news = list(brown.tagged_sents(categories='news'))
news = [[list(tup) for tup in sublist] for sublist in news]
for sublist in news:
    for x in sublist:
        x[1] = x[1].split(sep1, 1)[0]
        x[1] = x[1].split(sep2, 1)[0]




size = int(len(news) * 0.9)
training_data = news[:size]
test_data = news[size:]



# this adds the START and STOP tags to the beginning and end of sentence tags
training_tags_sents = [['START']+[x[1] for x in sublist]+['STOP'] for sublist in training_data]


all_words = list(set([x[0] for sentence in news for x in sentence]))
all_tags = ['START', 'STOP']+list(set([x[1] for sentence in news for x in sentence]))
all_bigrams = [[x,y] for x in all_tags for y in all_tags]
known_words = [x[0] for sublist in training_data for x in sublist]
training_tags_bigrams = [sentence[i:i + 2] for sentence in training_tags_sents for i in range(len(sentence) - 1)]


# now, take training data and get q and e probabilities


def q_mle():
    q = np.zeros((len(all_tags), len(all_tags)))
    bigrams_dict = dict(Counter([tuple(bigram) for bigram in training_tags_bigrams]))
    for bigram in bigrams_dict:
        # this is the MLE for q
        q[all_tags.index(bigram[0]), all_tags.index(bigram[1])] = bigrams_dict[bigram]
    for i in range(len(all_tags)):
        if np.any(q[i,:]):
            q[i,:] /= np.sum(q[i,:])
    return q


def e_mle():
    e = np.zeros((len(all_words), len(all_tags)))
    flattened_training_data = [x for sublist in training_data for x in sublist]
    all_tuples = dict(Counter([tuple(x) for x in flattened_training_data]))
    for tup in all_tuples:
        e[all_words.index(tup[0]), all_tags.index(tup[1])] = all_tuples[tup]
    for j in range(len(all_tags)):
        if np.any(e[:, j]):
            e[:, j] = (e[:,j] + 1)/(np.sum(e[:, j]) + len(all_words))
        else:
            e[:, j] = np.ones(len(all_words))/len(all_words)
    return e


q = q_mle()
e = e_mle()

def viterbi(x):
    n = len(x)
    p = np.zeros((n + 1, len(all_tags)))
    bp = np.zeros((n+1, len(all_tags)))
    p[0, all_tags.index('START')] = 1
    for k in np.arange(1, n + 1):
            p[k,:] = np.amax(np.expand_dims(p[k-1,:], axis=1)*q, axis=0)*e[k-1,:]
            bp[k, :] = np.argmax(np.expand_dims(p[k-1,:], axis=1)*q, axis=0)
    yn_index = np.argmax(p[n, :] * q[:, all_tags.index('STOP')])
    return getoptimaltags(bp, int(yn_index))


def getoptimaltags(bp, yn_index):
    y = [all_tags[yn_index]]
    n = bp.shape[0]-1
    pointer = int(yn_index)
    for i in np.arange(n, 1, -1):
        pointer = int(bp[i, pointer])
        y = [all_tags[pointer]] + y
    return y


def test_the_test_data():
    known_accuracy = 0
    known_size = 0
    unknown_accuracy = 0
    unknown_size = 0
    for sentence in test_data:
        test_words_in_sentence = [x[0] for x in sentence]
        viterbi_tags = viterbi(test_words_in_sentence)
        real_tags = [x[1] for x in sentence]
        for i in range(len(viterbi_tags)):
            if is_word_unknown(test_words_in_sentence[i]):
                unknown_size += 1
                if real_tags[i] == viterbi_tags[i]:
                    unknown_accuracy += 1
            else:
                known_size += 1
                if real_tags[i] == viterbi_tags[i]:
                    known_accuracy += 1
    return [known_accuracy, known_size, unknown_accuracy, unknown_size]


def is_word_unknown(w):
    if w in known_words:
        return False
    return True


[known_accuracy, known_size, unknown_accuracy, unknown_size] = test_the_test_data()
print("The error rate for known words is: "+str(1-known_accuracy/known_size)+", "+str(1-unknown_accuracy/unknown_size)+" for unknown words, and "+str(1-(known_accuracy+unknown_accuracy)/(known_size+unknown_size))+" for the total error rate.")