from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, TimeDistributed, Flatten, merge#, GlobalAveragePooling1D, regularizers
from keras.layers.embeddings import Embedding
from keras.layers import wrappers, Input, recurrent, InputLayer
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model, Sequential
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras import layers, regularizers, initializers

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import train_test_split

from SD_utils import *
from nmt_utils import *

import os
import io
import pprint as pp
import gc

# import torch
import pickle
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm_notebook, trange
import os
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
# from tools import *
# import convert_examples_to_features

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz-2.38/bin'

# word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d2.txt')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/vectors.txt')
#word_to_index_FT, index_to_word_FT, word_to_vec_map_FT = read_glove_vecs('data/wiki-news-300d-1M2.vec')

X, y = read_csv('data/SD_dataset_FINAL.csv')
m = len(y)
maxLen = len(max(X, key=len).split()) + 1

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    words = sentence.lower().split()
    avg = np.zeros(50)
    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
    avg = avg / len(words)

    return avg


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

    m = X_s.shape[0]  # number of training examples

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


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 2  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)
    print(emb_dim)
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def modelV1(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=500):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    np.random.seed(1)

    # Define number of training examples
    r = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 50  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(r):  # Loop over the training examples

            avg = sentence_to_avg(X[i], word_to_vec_map)

            z = np.dot(W, avg) + b
            a = softmax_SD(z)

            cost = -np.dot(Y_oh[i], np.log(a))

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred_ret = predict(X, Y, W, b, word_to_vec_map)

    return pred_ret, W, b


def modelV2(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(lay1_num, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(lay2_num, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model


def modelV3(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128):
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = Bidirectional(LSTM(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = Bidirectional(LSTM(lay2_num, return_sequences=False), input_shape=input_shape)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model


def modelV4(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128, n_features=50,
            isRandom=False, isAttention=True):
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')

    embeddings = None
    if isRandom:
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)  # embedding_layer(sentence_indices)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    else:
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)
    # X = Dropout(0.25)(embeddings)

    X = Bidirectional(LSTM(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)

    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    #   X = AttentionDecoder(lay2_num, n_features)
    if isAttention:
        attention = Dense(1, activation='tanh')(X)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(lay1_num * 2)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = merge.Multiply()([X, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

        X = Dropout(0.5)(sent_representation)
    else:
        X = Bidirectional(LSTM(lay2_num, return_sequences=False), input_shape=input_shape)(X)
        X = Dropout(0.5)(X)
        pass

    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model


def modelV5(input_shape=(maxLen,),
            lay1_num=128, lay2_num=128, n_features=50, embedding='glove'):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embeddings = None
    if embedding == 'random':
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)  # embedding_layer(sentence_indices)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    elif embedding == 'glove':
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)
    elif embedding == 'bert':
        pass
    # X = Dropout(0.25)(embeddings)
    X = Bidirectional(LSTM(lay1_num, return_sequences=True), input_shape=input_shape)(embeddings)
    X = Dropout(0.5)(X)
    attention = Dense(1, activation='tanh')(X)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lay1_num * 2)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = merge.Multiply()([X, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    X = Dropout(0.5)(sent_representation)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    #    attn_layer = AttentionLayer()

    # Create Model instance which converts sentence_indices into X.
    modelV5 = Model(inputs=sentence_indices, outputs=X)
    modelV5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(modelV5.summary())
    ### END CODE HERE ###

    return modelV5


def modelV6(input_shape=(maxLen,)):
    model = Sequential()
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map_FT, word_to_index_FT)
    embeddings = embedding_layer(sentence_indices)
    # model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, input_length=sequence_length, trainable=False))
    X = Dropout(0.5)(embeddings)
    X = GlobalAveragePooling1D()(X)
    X = Dense(2, activation='softmax')(X)
    modelV6 = Model(inputs=sentence_indices, outputs=X)
    return modelV6

class Conv1DBlock(layers.Layer):
    def __init__(self, filter_sizes, regularizer=regularizers.l2(0.01)):
        self.filter_sizes = filter_sizes
        self.regularizer = regularizer
        super(Conv1DBlock, self).__init__()
    def __call__(self, inputs, **kwargs):
        conv_layers = []
        for filter_size in self.filter_sizes:
            conv_layers.append(
                layers.BatchNormalization()(
                    layers.PReLU(alpha_initializer=initializers.constant(0.01),
                                alpha_regularizer=self.regularizer)(
                    layers.GlobalMaxPooling1D()(
                            layers.Conv1D(filters=filter_size[0],
                                         kernel_size=filter_size[1],
                                         padding="valid",
                                         strides=filter_size[2],
                                         kernel_regularizer=self.regularizer,
                                         bias_regularizer=self.regularizer)(inputs)))))
        return layers.Concatenate()(conv_layers)

def modelV8(input_shape, word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128, n_features=50,
            isRandom=False, isAttention=True, embedding='glove'):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embeddings = None
    if embedding == 'random':
        vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]

        embedding_layer = Embedding(len(word_to_index) + 1, 50,
                                    input_length=maxLen)  # embedding_layer(sentence_indices)
        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = np.random.rand(1, emb_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        embeddings = embedding_layer(sentence_indices)
    elif embedding == 'glove':
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)
    elif embedding == 'bert':
        pass
    # X = Dropout(0.25)(embeddings)
    X = Conv1DBlock([[lay1_num, 1, 1], [lay1_num, 2, 1], [lay1_num, 3, 1]])(embeddings)
    X = Dropout(0.5)(X)
    # attention = Dense(1, activation='tanh')(X)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(lay1_num * 2)(attention)
    # attention = Permute([2, 1])(attention)
    #
    # sent_representation = merge.Multiply()([X, attention])
    # sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    #
    # X = Dropout(0.5)(sent_representation)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2, activation='softmax')(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    #    attn_layer = AttentionLayer()

    # Create Model instance which converts sentence_indices into X.
    modelV8 = Model(inputs=sentence_indices, outputs=X)
    modelV8.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ### END CODE HERE ###

    return modelV8


tests_dict, wrongs_dict = {}, {}
num_it = 10
n_models = 8
acc_sum_v = [0] * (n_models + 1)
fp_sum_v, fn_sum_v = [0] * (n_models + 1), [0] * (n_models + 1)
tp_sum_v, tn_sum_v = [0] * (n_models + 1), [0] * (n_models + 1)
num_0s_sum, num_1s_sum = 0, 0

runModels = [None, False,
             False,
             False,
             True,
             False,
             False,
             False,
             False]

gb_accs = [0, 0, 0]
if __name__ == "__main__":
    for it in range(num_it):
        print("\nIteration #" + str(it + 1))

        X, y = read_csv('data/SD_dataset_FINAL.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random)
        maxLen = len(max(X, key=len).split()) + 1
        y_oh_train = convert_to_one_hot(y_train, C=2)
        y_oh_test = convert_to_one_hot(y_test, C=2)

        num_1s = sum(y_test)
        num_0s = y_test.shape[0] - num_1s
        num_0s_sum += num_0s
        num_1s_sum += num_1s

        # V1 model
        if runModels[1]:
            pred, W, b = modelV1(X_train, y_train, word_to_vec_map)
            pred_train, _ = predict(X_train, y_train, W, b, word_to_vec_map)
            predV1, acc1 = predict(X_test, y_test, W, b, word_to_vec_map)

            v1_tp, v1_tn, v1_fp, v1_fn = 0, 0, 0, 0
            for k in range(len(X_test)):
                num = predV1[k]
                if num != y_test[k]:
                    if int(num) == 1:
                        v1_fp += 1
                    elif int(num) == 0:
                        v1_fn += 1
                else:
                    if int(num) == 1:
                        v1_tp += 1
                    elif int(num) == 0:
                        v1_tn += 1
            tp_sum_v[1] += v1_tp
            tn_sum_v[1] += v1_tn
            fp_sum_v[1] += v1_fp
            fn_sum_v[1] += v1_fn
            acc_sum_v[1] += acc1

            print("0 misclass: " + str(v1_fp / (v1_tn + v1_fp)))
            print("1 misclass: " + str(v1_fn / (v1_tp + v1_fn)))
            print("Overall accuracy: " + str(acc1))
            gc.collect()

        # V2 model
        if runModels[2]:
            model_v2 = modelV2((maxLen,), word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128)
            model_v2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            model_v2.fit(X_train_indices, y_train_oh, epochs=30, batch_size=32, shuffle=True)

            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc2 = model_v2.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v2.predict(X_test_indices)

            v2_tp, v2_tn, v2_fp, v2_fn = 0, 0, 0, 0
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    # if X_test[k] in wrongs_dict:
                    #     wrongs_dict[X_test[k]] += 1
                    # else:
                    #     wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v2_fp += 1
                    elif int(num) == 0:
                        v2_fn += 1
                else:
                    if int(num) == 1:
                        v2_tp += 1
                    elif int(num) == 0:
                        v2_tn += 1
            tp_sum_v[2] += v2_tp
            tn_sum_v[2] += v2_tn
            fp_sum_v[2] += v2_fp
            fn_sum_v[2] += v2_fn

            print("0 misclass: " + str(v2_fp / (v2_tn + v2_fp)))
            print("1 misclass: " + str(v2_fn / (v2_tp + v2_fn)))
            print("Overall accuracy: " + str(acc2))
            acc_sum_v[2] += acc2
            gc.collect()

            plot_model(model_v2, to_file='model_v2.png')

        # V3 model
        if runModels[3]:
            v3_tp, v3_tn, v3_fp, v3_fn = 0, 0, 0, 0
            model_v3 = modelV3((maxLen,), word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128)
            model_v3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            model_v3.fit(X_train_indices, y_train_oh, epochs=25, batch_size=32, shuffle=True)

            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc3 = model_v3.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v3.predict(X_test_indices)
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    # if X_test[k] in wrongs_dict:
                    #     wrongs_dict[X_test[k]] += 1
                    # else:
                    #     wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v3_fp += 1
                    elif int(num) == 0:
                        v3_fn += 1
                else:
                    if int(num) == 1:
                        v3_tp += 1
                    elif int(num) == 0:
                        v3_tn += 1
            tp_sum_v[3] += v3_tp
            tn_sum_v[3] += v3_tn
            fp_sum_v[3] += v3_fp
            fn_sum_v[3] += v3_fn
            acc_sum_v[3] += acc3

            print("0 misclass: " + str(v3_fp / (v3_tn + v3_fp)))
            print("1 misclass: " + str(v3_fn / (v3_tp + v3_fn)))
            print("Overall accuracy: " + str(acc3))
            gc.collect()

            plot_model(model_v3, to_file='model_v3.png')

        # V4 Model
        if runModels[4]:
            v4_tp, v4_tn, v4_fp, v4_fn = 0, 0, 0, 0
            STAMP = 'simple_lstm_glove_vectors'
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            bst_model_path = STAMP + '.h5'
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

            model_v4 = modelV4((maxLen,), word_to_vec_map, word_to_index, lay1_num=128, lay2_num=128, n_features=maxLen,
                               isRandom=False, isAttention=True)
            model_v4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            #plot_model(model_v4, to_file='model_v4.png')

            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            # print(y_train_oh.shape)
            model_v4.fit(X_train_indices, y_train_oh, epochs=30, batch_size=32, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint])

            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc4 = model_v4.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v4.predict(X_test_indices)
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    if X_test[k] in wrongs_dict:
                        wrongs_dict[X_test[k]] += 1
                    else:
                        wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v4_fp += 1
                    elif int(num) == 0:
                        v4_fn += 1
                else:
                    if int(num) == 1:
                        v4_tp += 1
                    elif int(num) == 0:
                        v4_tn += 1
            tp_sum_v[4] += v4_tp
            tn_sum_v[4] += v4_tn
            fp_sum_v[4] += v4_fp
            fn_sum_v[4] += v4_fn
            acc_sum_v[4] += acc4

            print("0 misclass: " + str(v4_fp / (v4_tn + v4_fp)))
            print("1 misclass: " + str(v4_fn / (v4_tp + v4_fn)))
            print("Overall accuracy: " + str(acc4))



        # V5 Model
        if runModels[5]:
            v5_tp, v5_tn, v5_fp, v5_fn = 0, 0, 0, 0
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            model_v5 = modelV5((maxLen,), word_to_vec_map, word_to_index, lay1_num=128, lay2_num=64, n_features=maxLen)
            model_v5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_v5.fit(X_train_indices, y_train_oh, epochs=10, batch_size=32, shuffle=True)

            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc5 = model_v5.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v5.predict(X_test_indices)
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    if X_test[k] in wrongs_dict:
                        wrongs_dict[X_test[k]] += 1
                    else:
                        wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v5_fp += 1
                    elif int(num) == 0:
                        v5_fn += 1
                else:
                    if int(num) == 1:
                        v5_tp += 1
                    elif int(num) == 0:
                        v5_tn += 1
            tp_sum_v[5] += v5_tp
            tn_sum_v[5] += v5_tn
            fp_sum_v[5] += v5_fp
            fn_sum_v[5] += v5_fn
            acc_sum_v[5] += acc5

            print("0 misclass: " + str(v5_fp / (v5_tn + v5_fp)))
            print("1 misclass: " + str(v5_fn / (v5_tp + v5_fn)))
            print("Overall accuracy: " + str(acc5))
            trained_embeddings = model_v5.layers[1].get_weights()
            print(trained_embeddings)
            print(trained_embeddings[0].shape)

            scale = StandardScaler()
            X_train_s = scale.fit_transform(X_train_indices)

            params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                      'min_samples_split': 5}

            ann_estimator = KerasClassifier(build_fn=modelV5, epochs=10, batch_size=32, verbose=3)
            boosted_ann = AdaBoostClassifier(base_estimator=ann_estimator, random_state=1)
            print(X_train_s.shape)
            print(X_train_s)
            boosted_ann.fit(X_train_s, y_train.ravel())  # scale your training data
            boosted_ann.predict(X_test.reshape(-1, 1))
            vocab_len, emb_dim = len(word_to_index) + 1, word_to_vec_map["cucumber"].shape[0]
            # X_train_matrix = np.zeros((len(X_train), emb_dim))
            # # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
            # for word, index in word_to_index.items():
            #     X_train_matrix[index, :] = word_to_vec_map[word]

        if runModels[6]:
            v6_tp, v6_tn, v6_fp, v6_fn = 0, 0, 0, 0
            model_v6 = modelV6((maxLen,))
            model_v6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            # print(y_train_oh.shape)
            model_v6.fit(X_train_indices, y_train_oh, epochs=500, batch_size=32, shuffle=True)
            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc6 = model_v6.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v6.predict(X_test_indices)
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    if X_test[k] in wrongs_dict:
                        wrongs_dict[X_test[k]] += 1
                    else:
                        wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v6_fp += 1
                    elif int(num) == 0:
                        v6_fn += 1
                else:
                    if int(num) == 1:
                        v6_tp += 1
                    elif int(num) == 0:
                        v6_tn += 1
            tp_sum_v[6] += v6_tp
            tn_sum_v[6] += v6_tn
            fp_sum_v[6] += v6_fp
            fn_sum_v[6] += v6_fn
            acc_sum_v[6] += acc6

            print("0 misclass: " + str(v6_fp / (v6_tn + v6_fp)))
            print("1 misclass: " + str(v6_fn / (v6_tp + v6_fn)))
            print("Overall accuracy: " + str(acc6))
            # plot_model(model_v6, to_file='model_v6.png')

        if runModels[7]:
            X_train_avg = np.zeros((X_train.shape[0], 50))
            X_test_avg = np.zeros((X_test.shape[0], 50))
            for i in range(X_train.shape[0]):
                X_train_avg[i,] = sentence_to_avg(X_train[i,], word_to_vec_map)
            for j in range(X_test.shape[0]):
                X_test_avg[j] = sentence_to_avg(X_test[j], word_to_vec_map)

            gbc = GradientBoostingClassifier(random_state=0, max_depth=5, min_samples_split=2, n_estimators=3000)
            gbc.fit(X_train_avg, y_train)
            gb_acc = gbc.score(X_test_avg, y_test)
            print(gb_acc)

            y_pred = gbc.predict(X_test_avg)
            gb_accs[2] += f1_score(y_test, y_pred)
            gb_accs[0] += precision_score(y_test, y_pred)
            gb_accs[1] += recall_score(y_test, y_pred)

            print(gb_accs[0] / (it + 1))
            print(gb_accs[1] / (it + 1))
            print(gb_accs[2] / (it + 1))

        if runModels[8]:
            v8_tp, v8_tn, v8_fp, v8_fn = 0, 0, 0, 0
            X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
            y_train_oh = convert_to_one_hot(y_train, C=2)
            model_v8 = modelV8((maxLen,), word_to_vec_map, word_to_index, lay1_num=256, lay2_num=64, n_features=maxLen)
            model_v8.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_v8.fit(X_train_indices, y_train_oh, epochs=25, batch_size=32, shuffle=True)

            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
            y_test_oh = convert_to_one_hot(y_test, C=2)
            loss, acc8 = model_v8.evaluate(X_test_indices, y_test_oh)
            predV2 = model_v8.predict(X_test_indices)
            for k in range(len(X_test)):
                num = np.argmax(predV2[k])
                if num != y_test[k]:
                    if X_test[k] in wrongs_dict:
                        wrongs_dict[X_test[k]] += 1
                    else:
                        wrongs_dict[X_test[k]] = 1
                    if int(num) == 1:
                        v8_fp += 1
                    elif int(num) == 0:
                        v8_fn += 1
                else:
                    if int(num) == 1:
                        v8_tp += 1
                    elif int(num) == 0:
                        v8_tn += 1
            tp_sum_v[8] += v8_tp
            tn_sum_v[8] += v8_tn
            fp_sum_v[8] += v8_fp
            fn_sum_v[8] += v8_fn
            acc_sum_v[8] += acc8

            print("0 misclass: " + str(v8_fp / (v8_tn + v8_fp)))
            print("1 misclass: " + str(v8_fn / (v8_tp + v8_fn)))
            print("Overall accuracy: " + str(acc8))

        print("================================================================================")
        acc_avg_v = np.divide(acc_sum_v, float(it + 1))
        v_num = len(acc_sum_v)
        wrongs_dict_worst = dict((k, v) for k, v in wrongs_dict.items() if v >= (int((it + 1) / 2) + 1))
        pp.pprint(wrongs_dict_worst)
        for v in range(1, v_num):
            if not (tp_sum_v[v] == 0 or fp_sum_v == 0 or fn_sum_v == 0):
                p = tp_sum_v[v] / (tp_sum_v[v] + fp_sum_v[v])
                r = tp_sum_v[v] / (tp_sum_v[v] + fn_sum_v[v])
                f1 = (2 * p * r) / (p + r)
                print("=============== MODEL V" + str(v) + " =====================")
                print("V" + str(v) + " Accuracy: " + str(acc_avg_v[v]))

                print("Total Number of 0 labels: " + str(num_0s_sum) +
                      "\tNumber Mislabeled: " + str(fp_sum_v[v]) +
                      "\tMislabel Rate: " + str(fp_sum_v[v] / num_0s_sum))
                print("Total Number of 1 labels: " + str(num_1s_sum) +
                      "\tNumber Mislabeled: " + str(fn_sum_v[v]) +
                      "\tMislabel Rate: " + str(fn_sum_v[v] / num_1s_sum))
                print("Precision: " + str(p))
                print("Recall: " + str(r))
                print("F1 Score: " + str(f1))
