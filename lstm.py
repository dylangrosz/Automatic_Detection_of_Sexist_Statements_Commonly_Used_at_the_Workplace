from SexisteDetectionMain import pretrained_embedding_layer
from data_handler import get_data
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM, Bidirectional, RepeatVector, Permute, merge, Lambda
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, \
    GlobalMaxPooling1D
import keras.backend as K
import numpy as np
import pandas as pd
import pdb, json
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, \
    precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
import codecs
from sklearn.utils import shuffle
import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
from batch_gen import batch_gen
import sys
from SD_utils import *
from nltk import tokenize as tokenize_nltk

from my_tokenizer import glove_tokenize
from SD_utils import read_glove_vecs

EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 512
SCALE_LOSS_FUN = None

vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}

word2vec_model = None


def sentences_to_indices(X_s, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = len(X_s)  # number of training examples
    print(m)

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))  # If there is more than one dimension use ()

    for i in range(m):  # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X_s[i].lower().split()

        j = 0
        for w in sentence_words:
            if j < max_len and w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            j = j + 1

    ### END CODE HERE ###

    return X_indices


def get_embedding(word):
    # return
    try:
        return word2vec_model[word]
    except Exception as e:
        return np.zeros(EMBEDDING_DIM)


def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    return embedding


def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = TOKENIZER(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:  # Not a blank tweet
            tweet_return.append(tweet)
    # pdb.set_trace()
    return tweet_return


def gen_vocab(tweets):
    # Processing
    vocab_index = 1
    for tweet in tweets:
        text = TOKENIZER(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        #words = [word for word in words if word not in STOPWORDS]
        #print(words)
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word  # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    vocab['bsss'] = len(vocab) + 2
    reverse_vocab[len(vocab)] = 'UNK'
    reverse_vocab[len(vocab) + 1] = 'bsss'


def filter_vocab(k):
    global freq, vocab
    pdb.set_trace()
    freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence(tweets, y):
    X = []
    for tweet in tweets:
        text = TOKENIZER(tweet.lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        # words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            if word in word_to_index:
                seq.append(word_to_index[word])
        X.append(seq)
    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    sentence_indices = Input(shape=sequence_length, dtype='int32')

    seq = Sequential()
    embeddings = Embedding(len(vocab) + 2, embedding_dim, input_length=sequence_length[0], trainable=LEARN_EMBEDDINGS)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = Bidirectional(LSTM(64, return_sequences=True), input_shape=sequence_length)(embeddings)
    X = Dropout(0.5)(X)
    # X = Bidirectional(LSTM(64, return_sequences=False), input_shape=sequence_length)(X)
    attention = Dense(1, activation='tanh')(X)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)


    sent_representation = merge.Multiply()([X, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    X = Dropout(0.5)(sent_representation)
    X = Dense(2, activation='softmax')(X)
    X = Activation('softmax')(X)
    model_lstm = Model(inputs=sentence_indices, outputs=X)
    model_lstm.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])

    print(model_lstm.summary())
    return model_lstm

def train_LSTM(X, y, model, inp_dim, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    print(NO_OF_FOLDS)
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[1].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0] / float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0] / float(len(y_temp))

                try:
                    y_temp = convert_to_one_hot(y_temp, C=2)
                except Exception as e:
                    print(e)
                    print(y_temp)
                # print(x.shape)
                # print(y_temp.shape)
                # print("HERE WE GO")
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred))
        print(y_pred)
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    print("macro results are")
    print("average precision is " + str(p / NO_OF_FOLDS))
    print("average recall is " + str(r / NO_OF_FOLDS))
    print("average f1 is " + str(f1 / NO_OF_FOLDS))

    print("micro results are")
    print("average precision is " + str(p1 / NO_OF_FOLDS))
    print("average recall is " + str(r1 / NO_OF_FOLDS))
    print("average f1 is " + str(f11 / NO_OF_FOLDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)

    args = parser.parse_args()
    GLOVE_MODEL_FILE = str(args.embeddingfile)
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    KERNEL = args.kernel
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function

    np.random.seed(SEED)
    print('GLOVE embedding: ' + str(GLOVE_MODEL_FILE))
    print('Embedding Dimension: ' + str(int(EMBEDDING_DIM)))
    print('Allowing embedding learning: ' + str(LEARN_EMBEDDINGS))

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d2.txt')
    word2vec_model = word_to_vec_map.copy()
    X_tweets, y = read_csv('data/SD_dataset_FINAL.csv')

    print(np.array(word_to_vec_map.keys()).shape)

#    gen_vocab(X_tweets)
    vocab = word_to_index.copy()
    reverse_vocab = index_to_word.copy()
    vocab['UNK'] = len(vocab) + 1
    print(len(vocab))
    reverse_vocab[len(vocab)] = 'UNK'

    X, y = gen_sequence(X_tweets, y)
    MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x), X))
    print("max seq length is " + str(MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = shuffle(data, y)
    W = get_embedding_weights()
    # print(data.shape[1])
    model = lstm_model((data.shape[1],), EMBEDDING_DIM)
    # model = lstm_model(data.shape[1], 25, get_embedding_weights())
    # y_oh = convert_to_one_hot(y, C=2)

    # print(data)
    # print(y)
    # print(data.shape)
    # print(y.shape)
    train_LSTM(data, y, model, EMBEDDING_DIM, W)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=np.random)
    # maxLen = MAX_SEQUENCE_LENGTH
    # print(X_train)
    # print(X_train.shape)
    # X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    y_train_oh = convert_to_one_hot(y_train, C=2)
    # print(np.array(X_train).shape)
    # print(y_train_oh.shape)
    # model.fit(X_train, y_train_oh, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    y_test_oh = convert_to_one_hot(y_test, C=2)
    loss, acc = model.evaluate(X_test, y_test_oh)
    pred = model.predict(X_test)
    table = model.layers[1].get_weights()[0]
    print(acc)
    print(table)
    print(np.array(table).shape)
    np.save('lstm_embed.npy', np.array(table))
    f_vocab = open('vocab_text', 'w')
    json.dump(f_vocab, 'vocab_text')

    pdb.set_trace()
