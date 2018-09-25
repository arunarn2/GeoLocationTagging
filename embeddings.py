import tensorflow as tf
import os
import pickle
from gensim.models import KeyedVectors as word2vec
import numpy as np
from os import path


def assign_pretrained_word_embedding(sess, args, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    word2vec_model = word2vec.load_word2vec_format(word2vec_model_path, binary=False, encoding="ISO-8859-1")
    word2vec_dict = {}
    for i in range(len(word2vec_model.wv.vocab)):
        word = word2vec_model.wv.index2word[i]
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        if embedding_vector is not None:
            # print word
            word2vec_dict[word] = embedding_vector
    # create an empty word_embedding list.
    word_embedding_2dlist = [[]] * vocab_size
    # assign empty for first word:'PAD'
    word_embedding_2dlist[0] = np.zeros(args.embed_size)
    # bound for random variables.
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist = 0
    count_not_exist = 0
    # loop each word
    for i in range(1, vocab_size):
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            # try to get vector:it is an array.
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, args.embed_size);
            # init a random value for the word
            count_not_exist = count_not_exist + 1
    # covert to 2d array.
    word_embedding_final = np.array(word_embedding_2dlist)
    # convert to tensor
    print "word_embedding_final", word_embedding_final.shape, type(word_embedding_final)
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)

    # assign this value to our embedding variables of our model.
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ; word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


def create_vocabulary(simple=None, word2vec_model_path='glove.6B.300d.word2vec.txt', name_scope=''):
    if not path.exists("./cache_vocabulary_label_pik"):
        os.mkdir("./cache_vocabulary_label_pik")
    cache_path = 'cache_vocabulary_label_pik/' + name_scope + "_word_vocabulary.pik"
    print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        if simple is not None:
            word2vec_model_path = 'glove.6B.300d.word2vec.txt'
        print("create vocabulary. word2vec_model_path:", word2vec_model_path)
        model = word2vec.load_word2vec_format(word2vec_model_path, binary=False)  # encoding="ISO-8859-1"
        vocabulary_word2index['PAD_ID'] = 0
        vocabulary_index2word[0] = 'PAD_ID'
        special_index = 0
        if 'biLstmTextRelation' in name_scope:
            # a special token for biLstTextRelation model. which is used between two sentences.
            vocabulary_word2index['EOS'] = 1
            vocabulary_index2word[1] = "EOS"
            special_index = 1
        for i, vocab in enumerate(model.vocab):
            vocabulary_word2index[vocab] = i + 1 + special_index
            vocabulary_index2word[i + 1 + special_index] = vocab

        # save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word
