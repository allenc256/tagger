import argparse
import types
import logging
import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tagger.model.model import Model, dump_statistics
from tagger.model.dataset import dataset, DatasetDictionary


def main():
    # argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train', metavar='FILE',
        default='data/wiki/parsed_examples.train.tfrecords.gz',
        help='training set tfrecords file')
    parser.add_argument(
        '--dev', metavar='FILE',
        default='data/wiki/parsed_examples.dev.tfrecords.gz',
        help='dev set tfrecords file')
    parser.add_argument(
        '--dict', metavar='FILE',
        default='data/wiki/parsed_examples.dict.json',
        help='data/feature dictionary file')
    parser.add_argument(
        '--word-embedding', metavar='FILE',
        default='data/wiki/parsed_examples.embedding.npy',
        help='word embedding file')
    parser.add_argument(
        '--hidden-dim', metavar='N', type=int, default=128,
        help='hidden dimension size')
    parser.add_argument(
        '--num-encoder-layers', metavar='N', type=int, default=3,
        help='number of encoder RNN layers')
    parser.add_argument(
        '--num-memory-layers', metavar='N', type=int, default=1,
        help='number of memory RNN layers')
    parser.add_argument(
        '--dropout-rate', metavar='F', type=float, default=0.2,
        help='dropout rate')
    parser.add_argument(
        '--tag-pos-weight', metavar='F', type=float, default=3.0,
        help='weighted cross entropy positive weight (higher numbers should '
             'lead to increased recall at the expense of precision)')
    parser.add_argument(
        '--learning-rate', metavar='F', type=float, default=1e-3,
        help='initial learning rate')
    parser.add_argument(
        '--grad-clip-norm', metavar='F', type=float, default=5.0,
        help='gradient clipping global norm')
    parser.add_argument(
        '--log', metavar='FILE', default='logs/train.log',
        help='path to log file')
    parser.add_argument(
        '--max-len', metavar='N', type=int, default=120,
        help='maximum example length')
    parser.add_argument(
        '--batch-size', metavar='N', type=int, default=64,
        help='minibatch size')
    parser.add_argument(
        '--limit', metavar='N', type=int, help='limit on number of examples')
    parser.add_argument(
        '--shuffle-size', metavar='N', type=int, default=1000,
        help='shuffle buffer size')
    parser.add_argument(
        '--train-steps', metavar='N', type=int, default=100000,
        help='number of training steps')
    parser.add_argument(
        '--eval-steps', metavar='N', type=int, default=100,
        help='number of evaluation steps')
    parser.add_argument(
        '--eval-interval', metavar='N', type=int, default=1000,
        help='number of steps between evaluations')
    parser.add_argument(
        '--debug', action='store_true', help='debug mode')
    embedding_dims = types.SimpleNamespace(
        ent_type=16, is_title=2, like_num=2, pos=16, tag=16)
    for name, dim in embedding_dims.__dict__.items():
        parser.add_argument(
            '--embedding-dim-%s' % name.replace('_', '-'), type=int,
            metavar='N', default=dim,
            help='embedding dimension for feature %s' % name)
    args = parser.parse_args()

    # set parsed embedding dims
    hparams = types.SimpleNamespace(**args.__dict__)
    for name in embedding_dims.__dict__.keys():
        dim = getattr(hparams, 'embedding_dim_%s' % name)
        delattr(hparams, 'embedding_dim_%s' % name)
        setattr(embedding_dims, name, dim)
    hparams.embedding_dims = embedding_dims

    # config logging
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(
        filename=args.log, level=logging.DEBUG,
        format='%(asctime)s - - %(levelname)s - %(message)s')

    run(hparams)


def run(hparams):
    logging.info('hyper parameters: %s', hparams)

    # load embedding
    with open(hparams.word_embedding, 'rb') as f:
        word_embedding = np.load(f)

    # load dict
    with open(hparams.dict, 'rt') as f:
        data_dict = DatasetDictionary()
        data_dict.from_json(json.load(f))

    with tf.Session() as sess:
        # load datasets
        data_train = dataset(
            hparams.train, hparams.max_len, compression_type='GZIP',
            batch_size=hparams.batch_size, limit=hparams.limit,
            shuffle_size=hparams.shuffle_size, repeat=True)
        data_dev = dataset(
            hparams.dev, hparams.max_len, compression_type='GZIP',
            batch_size=hparams.batch_size, limit=hparams.limit,
            shuffle_size=hparams.shuffle_size, repeat=True)

        # prepare data iterator
        handle = tf.placeholder(tf.string, [])
        handle_train = data_train.make_one_shot_iterator().string_handle().eval()
        handle_dev = data_dev.make_one_shot_iterator().string_handle().eval()
        data_it = tf.data.Iterator.from_string_handle(
            handle, data_train.output_types, data_train.output_shapes)

        # initialize model
        model = Model(hparams, data_it, handle, data_dict, word_embedding)
        dump_statistics()

        # train
        sess.run(tf.global_variables_initializer())
        progress = tqdm(range(hparams.train_steps), desc='training')
        for i in progress:
            l, _, s = sess.run(
                [model.mean_loss, model.train_op, model.global_step],
                feed_dict={model.training: True, model.handle: handle_train})
            progress.set_postfix(loss=l, step=s)

            if (i+1) % hparams.eval_steps == 0:
                model.eval(sess, handle_train)


if __name__ == '__main__':
    main()
