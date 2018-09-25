import argparse


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments: argv -- An argument list without the program name.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dataset', metavar='str', help='dataset for dialectology', type=str, default='cmu')
    parser.add_argument('-task', '--task', metavar='str', help='regression, classify_states or classify_regions',
                        type=str, default='classify_states')
    parser.add_argument('-is_classifier', '--is_classifier', metavar='bool', help='Classification or Regression',
                        type=bool, default=True)
    parser.add_argument('-modelname', '--modelname', metavar='str', help='TextCNN, TextRNN or FastText', type=str,
                        default='TextCNN')
    parser.add_argument('-num_classes', '--num_classes', metavar='int', help='number of classes', type=int, default=49)
    parser.add_argument('-num_filters', '--num_filters', metavar='int', help='number of filters', type=int, default=256)
    parser.add_argument('-learning_rate', '--learning_rate', metavar='int', help='learning rate', type=float,
                        default=0.001)
    parser.add_argument('-batch_size', '--batch_size', metavar='int', help='SGD batch size', type=int, default=32)

    parser.add_argument('-decay_steps', '--decay_steps', metavar='int',
                        help='how many steps before decay learning rate', type=int, default=6000)

    parser.add_argument('-decay_rate', '--decay_rate', metavar='float', help='Rate of decay for learning rate.',
                        type=float, default=0.65)

    parser.add_argument('-batchnorm', '--batchnorm', metavar='bool', help='batchnorm', type=bool, default=True)
    parser.add_argument('-earlystop', '--earlystop', metavar='bool', help='earlystop', type=bool, default=True)

    parser.add_argument('-is_training', '--is_training', metavar='bool', help='Is training or testing', type=bool,
                        default=True)
    parser.add_argument('-num_epochs', '--num_epochs', metavar='int', help='Number of epochs to run', type=int,
                        default=100)
    parser.add_argument('-sentence_len', '--sentence_len', metavar='int', help='Max sentence length', type=bool,
                        default=5000)
    parser.add_argument('-use_embedding', '--use_embedding', metavar='bool', help='Use embedding', type=bool,
                        default=True)
    parser.add_argument('-validate_every', '--validate_every', metavar='int', help='Validate every num steps', type=int,
                        default=10)

    parser.add_argument('-traning_data_path', '--traning_data_path', metavar='str', help='traning_data_path', type=str,
                        default='./datasets/cmu')

    parser.add_argument('-word2vec_model_path', '--word2vec_model_path', metavar='str',
                        help='word2vecs vocabulary and vectors', type=str, default='glove.6B.300d.word2vec.txt')

    parser.add_argument('-embed_size', '--embed_size', metavar='int', help='embedding size', type=int, default=300)

    parser.add_argument('-hidden', '--hidden', metavar='int', help='Hidden layer size', type=int, default=300)
    parser.add_argument('-mindf', '--mindf', metavar='int', help='minimum document frequency in BoW', type=int,
                        default=10)
    parser.add_argument('-d', '--dir', metavar='str', help='home directory', type=str, default='./datasets/cmu')
    parser.add_argument('-enc', '--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str,
                        default='latin1')
    parser.add_argument('-reg', '--regularization', metavar='float', help='regularization coefficient)', type=float,
                        default=1e-6)
    parser.add_argument('-drop', '--dropout', metavar='float', help='dropout coef default 0.5', type=float, default=0.5)
    parser.add_argument('-optimizer', '--optimizer', type=str, help='Optimizer used for the neural network',
                        default='Adam')

    args = parser.parse_args(argv)
    if args.task == "classify_regions":
        args.num_classes = 4
        args.classifier = True
    elif args.task == "classify_states":
        args.num_classes = 49
        args.is_classifier = True
    else:
        args.num_classes = 2
        args.is_classifier = False
    return args
