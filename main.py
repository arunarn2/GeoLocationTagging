# -*- coding: utf-8 -*-
"""
Given a training set of lat/lon as input and probability distribution over words as output, train a
 model that can predict words based on location. Then try to visualise borders and regions (e.g.
 try many lat/lon as input and get the probability of word yinz in the output and visualise that).
"""

import logging
import os
import sys
from os import path

import numpy as np
import tensorflow as tf

import config
import data
import embeddings
from models.BiLstmTextRelation import BiLstmTextRelation
from models.FastText import FastText
from models.HierarchicalAttention import HierarchicalAttention
from models.Seq2SeqAttn import Seq2SeqAttn
from models.TextCNN import TextCNN
from models.TextRCNN import TextRCNN
from models.TextRNN import TextRNN

filter_sizes = [1, 2, 3, 4, 5, 6, 7]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Model Evaluation
def do_eval(sess, model, evalX, evalY, batch_size):
    number_examples = len(evalX)
    for start1, end1 in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        if args.modelname == "FastText":
            feed_dict1 = {model.input_x: evalX[start:end], model.input_y: evalY[start:end]}
        else:
            feed_dict1 = {model.input_x: evalX[start1:end1], model.dropout_keep_prob: 1}
            feed_dict1[model.input_y] = evalY[start1:end1]

        # curr_eval_acc--->model.accuracy
        curr_eval_loss, logits, curr_eval_acc = sess.run([model.loss_val, model.logits,
                                                          model.accuracy], feed_dict1)

        val_loss.append(curr_eval_loss)
        val_acc.append(curr_eval_acc)

    return np.mean(val_loss), np.mean(val_acc)


if __name__ == '__main__':
    ckpt_dir = "ckpt_dir"
    if not path.exists("./ckpt_dir"):
        os.mkdir("./ckpt_dir")

    # Parse Arguments
    args = config.parse_args(sys.argv[1:])
    datadir = args.dir

    # Load data
    dataset_name = 'cmu' if 'cmu' in datadir else 'na'
    logging.info('dataset: %s' % dataset_name)
    data = data.load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, dataset_name=dataset_name,
                          task=args.task)

    trainX, trainY, devX, devY, testX, testY, trainU, devU, testU, labels = data

    # Load embeddings
    vocabulary_word2index, vocabulary_index2word = embeddings.create_vocabulary(
        word2vec_model_path=args.word2vec_model_path, name_scope="cnn2")  # simple='simple'
    vocab_size = len(vocabulary_word2index)
    logging.info("cnn_model.vocab_size: %s" % vocab_size)

    # Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.info("Initializing model: %s" % args.modelname)
    max_checks_without_progress = 10
    checks_without_progress = 0
    best_loss = np.infty

    with tf.Session(config=config) as sess:
        # Used to determine when to stop the training early
        valid_loss_summary = []
        stop_early = 0

        if args.modelname == "TextCNN":
            # Instantiate Text CNN Model
            model = TextCNN(filter_sizes, num_filters=args.num_filters, num_classes=args.num_classes,
                            learning_rate=args.learning_rate, batch_size=args.batch_size, batchnorm=args.batchnorm,
                            decay_steps=args.decay_steps, decay_rate=args.decay_rate, sequence_length=args.sentence_len,
                            vocab_size=vocab_size, embed_size=args.embed_size, is_training=args.is_training,
                            is_classifier=args.is_classifier, optimizer=args.optimizer, clip_gradients=5.0,
                            decay_rate_big=0.5, initializer=tf.truncated_normal_initializer(stddev=0.1))
        elif args.modelname == "TextRNN":
            # Instantiate Text RNN Model
            model = TextRNN(num_classes=args.num_classes, learning_rate=args.learning_rate, batch_size=args.batch_size,
                            decay_steps=args.decay_steps, decay_rate=args.decay_rate,
                            sequence_length=args.sentence_len, vocab_size=vocab_size, embed_size=args.embed_size,
                            is_training=args.is_training, batchnorm=args.batchnorm, is_classifier=args.is_classifier,
                            optimizer=args.optimizer, initializer=tf.random_normal_initializer(stddev=0.1))
        elif args.modelname == "TextRCNN":
            # Instantiate Text RCNN Model
            model = TextRCNN(num_classes=args.num_classes, learning_rate=args.learning_rate, batch_size=args.batch_size,
                             decay_steps=args.decay_steps, decay_rate=args.decay_rate,
                             sequence_length=args.sentence_len, vocab_size=vocab_size, embed_size=args.embed_size,
                             is_training=args.is_training, is_classifier=args.is_classifier, optimizer=args.optimizer,
                             batch_norm=args.batchnorm, initializer=tf.random_normal_initializer(stddev=0.1))
        elif args.modelname == "FastText":
            # Instantiate FastText Model
            model = FastText(num_classes=args.num_classes, learning_rate=args.learning_rate, batch_size=args.batch_size,
                             decay_steps=args.decay_steps, decay_rate=args.decay_rate, num_sampled=20,
                             sequence_length=args.sentence_len, vocab_size=vocab_size, embed_size=args.embed_size,
                             is_training=args.is_training, batchnorm=args.batchnorm, is_classifier=args.is_classifier,
                             optimizer=args.optimizer)
        elif args.modelname == "Seq2SeqAttn":
            # Instantiate Seq2SeqAttn Model
            model = Seq2SeqAttn(num_classes=args.num_classes, learning_rate=args.learning_rate,
                                batch_size=args.batch_size, decay_steps=args.decay_steps, decay_rate=args.decay_rate,
                                sequence_length=args.sentence_len, vocab_size=vocab_size, embed_size=args.embed_size,
                                hidden_size=args.hidden, is_training=args.is_training,
                                is_classifier=args.is_classifier, optimizer=args.optimizer,
                                decoder_sent_length=args.num_classes,
                                initializer=tf.random_normal_initializer(stddev=0.1),
                                clip_gradients=5.0, l2_lambda=0.0001)
        elif args.modelname == "BiLstmTextRelation":
            # Instantiate BiLstmTextRelation Model
            model = BiLstmTextRelation(num_classes=args.num_classes, learning_rate=args.learning_rate,
                                       batch_size=args.batch_size, decay_steps=args.decay_steps,
                                       decay_rate=args.decay_rate, sequence_length=args.sentence_len,
                                       vocab_size=vocab_size, embed_size=args.embed_size, is_training=args.is_training,
                                       is_classifier=args.is_classifier, optimizer=args.optimizer,
                                       initializer=tf.random_normal_initializer(stddev=0.1))
        elif args.modelname == "HierarchicalAttention":
            # Instantiate HierarchicalAttention
            model = HierarchicalAttention(num_classes=args.num_classes, learning_rate=args.learning_rate,
                                          batch_size=args.batch_size, decay_steps=args.decay_steps,
                                          decay_rate=args.decay_rate, sequence_length=args.sentence_len,
                                          num_sentences=5, vocab_size=vocab_size, embed_size=args.embed_size,
                                          hidden_size=args.hidden, is_training=args.is_training,
                                          is_classifier=args.is_classifier, optimizer=args.optimizer,
                                          need_sentence_level_attention_encoder_flag=True,
                                          initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0)
        else:
            # Instantiate default TextCNN Model -- default
            model = TextCNN(filter_sizes, num_filters=args.num_filters, num_classes=args.num_classes,
                            learning_rate=args.learning_rate, batch_size=args.batch_size, batchnorm=args.batchnorm,
                            decay_steps=args.decay_steps, decay_rate=args.decay_rate, sequence_length=args.sentence_len,
                            vocab_size=vocab_size, embed_size=args.embed_size, is_training=args.is_training,
                            is_classifier=args.is_classifier, optimizer=args.optimizer, clip_gradients=5.0,
                            decay_rate_big=0.5, initializer=tf.random_normal_initializer(stddev=0.1))

        # Initialize Saver
        saver = tf.train.Saver()
        if os.path.exists(ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            # load pre-trained word embedding
            if args.use_embedding:
                embeddings.assign_pretrained_word_embedding(sess, args, vocabulary_index2word, vocab_size,
                                                            model, word2vec_model_path=args.word2vec_model_path)
        curr_epoch = sess.run(model.epoch_step)

        # 3. Feed data & Train
        number_of_training_data = len(trainX)
        print("Feed data and Train")
        batch_size = args.batch_size
        for epoch in range(curr_epoch, args.num_epochs):
            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []

            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                if args.modelname == "FastText":
                    feed_dict = {model.input_x: trainX[start:end], model.input_y: trainY[start:end]}
                else:
                    feed_dict = {model.input_x: trainX[start:end], model.dropout_keep_prob: 0.5}
                    feed_dict[model.input_y] = trainY[start:end]

                # curr_acc--->model.accuracy
                curr_loss, curr_acc, _ = sess.run([model.loss_val, model.accuracy, model.train_op], feed_dict)
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc

                # Record the loss and accuracy of each training
                train_loss.append(loss)
                train_acc.append(acc)

                if counter % 50 == 0:
                    if args.is_classifier:
                        # Train Accuracy:%.4f  acc/float(counter)
                        print("Epoch %d\tCounter %d\tTrain Loss:%.4f\tTrain Accuracy:%.4f" %
                              (epoch, counter, loss / float(counter), acc / float(counter)))
                    else:
                        print("Epoch %d\tCounter %d\tTrain Loss:%.4f" % (epoch, counter, loss / float(counter)))

            # epoch increment
            logging.info("Incrementing epoch counter....")
            sess.run(model.epoch_increment)

            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc)

            # 4.validation
            logging.info("Evaluating on Dev set ....")
            avg_valid_loss, avg_valid_acc = do_eval(sess, model, devX, devY, batch_size)
            valid_loss_summary.append(avg_valid_loss)

            if args.is_classifier:
                # Print the progress of each epoch
                print("Epoch: {}/{}".format(epoch, args.num_epochs), "Dev Loss: {:.4f}".format(avg_valid_loss),
                      "Dev Acc: {:.4f}".format(avg_valid_acc))
            else:
                # Print the progress of each epoch
                print("Epoch: {}/{}".format(epoch, args.num_epochs), "Dev Loss: {:.4f}".format(avg_valid_loss))

            if args.earlystop:
                if avg_valid_loss < best_loss:
                    save_path = ckpt_dir + "/model.ckpt"
                    logging.info("Saving model to checkpoint to %s" % save_path)
                    saver.save(sess, save_path, global_step=epoch)
                    best_loss = avg_valid_loss
                    checks_without_progress = 0
                else:
                    checks_without_progress += 1
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
            else:
                logging.info("Not using early stopping")

        # 5. Test
        logging.info("Running Test ....")
        test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size)
        if args.is_classifier:
            print("Test Loss: {:.4f}".format(test_loss), "Test Accuracy: {:.4f}".format(test_acc))
        else:
            print("Test Loss: {:.4f}".format(test_loss))
    pass
