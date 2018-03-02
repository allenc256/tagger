import argparse
import json
import logging
import os
import types
import shutil

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tagger.model.dataset as tagger_dataset
import tagger.model.model as tagger_model


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
        '--num-parallel-calls', metavar='N', type=int, default=2,
        help='number of parallel calls in dataset parser')
    parser.add_argument(
        '--dropout-rate', metavar='F', type=float, default=0.2,
        help='dropout rate')
    parser.add_argument(
        '--tag-pos-weight', metavar='F', type=float, default=2.0,
        help='weighted cross entropy positive weight (higher numbers should '
             'lead to increased recall at the expense of precision)')
    parser.add_argument(
        '--learning-rate', metavar='F', type=float, default=1e-3,
        help='initial learning rate')
    parser.add_argument(
        '--grad-clip-norm', metavar='F', type=float, default=5.0,
        help='gradient clipping global norm')
    parser.add_argument(
        '--logs', metavar='DIR', default='logs',
        help='path to log directory')
    parser.add_argument(
        '--max-len', metavar='N', type=int, default=120,
        help='maximum example length')
    parser.add_argument(
        '--batch-size', metavar='N', type=int, default=64,
        help='minibatch size')
    parser.add_argument(
        '--prefetch-size', metavar='N', type=int, default=64,
        help='prefetch buffer size')
    parser.add_argument(
        '--limit', metavar='N', type=int, help='limit on number of examples')
    parser.add_argument(
        '--shuffle-size', metavar='N', type=int, default=1000,
        help='shuffle buffer size')
    parser.add_argument(
        '--train-steps', metavar='N', type=int, default=200000,
        help='number of training steps')
    parser.add_argument(
        '--eval-steps', metavar='N', type=int, default=500,
        help='number of evaluation steps')
    parser.add_argument(
        '--eval-interval', metavar='N', type=int, default=10000,
        help='number of steps between evaluations')
    parser.add_argument(
        '--dump-examples', metavar='N', type=int, default=0,
        help='dump N examples during evaluation to log')
    parser.add_argument(
        '--restore', action='store_true',
        help='restore model from latest checkpoint')
    parser.add_argument(
        '--max-patience', metavar='N', type=int, default=3,
        help='maximum number of lower-performing evaluation steps before '
             'applying learning rate decay')
    parser.add_argument(
        '--model', metavar='TYPE', choices=['rnn', 'attention'],
        default='attention',
        help='model type to use (choices: rnn, attention)')

    group = parser.add_argument_group('hyperparameters for rnn model')
    group.add_argument(
        '--num-encoder-rnn-layers', metavar='N', type=int, default=3,
        help='number of encoder RNN layers')
    group.add_argument(
        '--num-memory-rnn-layers', metavar='N', type=int, default=1,
        help='number of memory RNN layers')

    group = parser.add_argument_group('hyperparameters for attention model')
    group.add_argument(
        '--num-attention-layers', metavar='N', type=int, default=3,
        help='number of self-attention layers')
    group.add_argument(
        '--hidden-dim-ff', metavar='N', type=int, default=128*2,
        help='feed-forward hidden dimension size')

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
    os.makedirs(args.logs, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.logs, 'train.log'), level=logging.DEBUG,
        format='%(asctime)s - - %(levelname)s - %(message)s')

    run(hparams)


def write_summ(summ_writer, tag, value, step):
    summ_writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), step)


def run(hparams):
    logging.info('hyper parameters: %s', hparams)

    # load embedding
    with open(hparams.word_embedding, 'rb') as f:
        word_embedding = np.load(f)

    # load dict
    with open(hparams.dict, 'rt') as f:
        data_dict = tagger_dataset.DatasetDictionary()
        data_dict.from_json(json.load(f))

    # summary dir
    summ_dir = os.path.join(hparams.logs, 'event')
    if not hparams.restore:
        logging.info('clearing summary directory: %s', summ_dir)
        shutil.rmtree(summ_dir, ignore_errors=True)
    os.makedirs(summ_dir, exist_ok=True)

    # save dir
    save_dir = os.path.join(hparams.logs, 'model')
    if not hparams.restore:
        logging.info('clearing model checkpoint directory: %s', save_dir)
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    with tf.Session() as sess, tf.summary.FileWriter(summ_dir) as sfw:
        # load datasets
        data_train = tagger_dataset.dataset(
            hparams.train, hparams.max_len, compression_type='GZIP',
            batch_size=hparams.batch_size, limit=hparams.limit,
            shuffle_size=hparams.shuffle_size,
            prefetch_size=hparams.prefetch_size, repeat=True,
            num_parallel_calls=hparams.num_parallel_calls)
        data_dev = tagger_dataset.dataset(
            hparams.dev, hparams.max_len, compression_type='GZIP',
            batch_size=hparams.batch_size, limit=hparams.limit,
            shuffle_size=hparams.shuffle_size,
            prefetch_size=hparams.prefetch_size, repeat=True,
            num_parallel_calls=hparams.num_parallel_calls)

        # prepare data iterator
        handle = tf.placeholder(tf.string, [])
        handle_train = data_train.make_one_shot_iterator().string_handle().eval()
        handle_dev = data_dev.make_one_shot_iterator().string_handle().eval()
        data_it = tf.data.Iterator.from_string_handle(
            handle, data_train.output_types, data_train.output_shapes)

        # initialize model
        if hparams.model == 'attention':
            model = tagger_model.AttentionModel(
                hparams, data_it, handle, data_dict, word_embedding)
        elif hparams.model == 'rnn':
            model = tagger_model.RnnModel(
                hparams, data_it, handle, data_dict, word_embedding)
        else:
            raise ValueError('invalid model type: %s', hparams.model)
        tagger_model.dump_statistics()

        # saver
        saver = tf.train.Saver()

        # init or restore model
        if hparams.restore:
            ckpt = tf.train.latest_checkpoint(save_dir)
            logging.info('restoring checkpoint: %s', ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.global_variables_initializer())

        # learning rate decayer
        decayer = tagger_model.LearningRateDecayer(model, hparams.max_patience)

        # train
        progress = tqdm(range(hparams.train_steps), desc='training')
        for i in progress:
            l, _, s = sess.run(
                [model.mean_loss, model.train_op, model.global_step],
                feed_dict={model.training: True, model.handle: handle_train})
            progress.set_postfix(loss=l, step=s)

            # evaluate
            if (i+1) % hparams.eval_interval == 0:
                f1_train, acc_train, l1_train, l2_train = tagger_model.evaluate(
                    sess, model, handle_train, header='train')
                f1_dev, acc_dev, l1_dev, l2_dev = tagger_model.evaluate(
                    sess, model, handle_dev, header='dev')
                l_train = l1_train + l2_train
                l_dev = l1_dev + l2_dev
                logging.info(
                    'training: step=%d, loss_train=%g (%g+%g), f1_train=%g, '
                    'acc_train=%g, loss_dev=%g (%g+%g), f1_dev=%g, acc_dev=%g',
                    s, l_train, l1_train, l2_train, f1_train, acc_train,
                    l_dev, l1_dev, l2_dev, f1_dev, acc_dev)

                # apply learning rate decay
                decayer.decay(l_dev)

                # write eval summaries
                write_summ(sfw, 'loss/train', l_train, s)
                write_summ(sfw, 'loss1/train', l1_train, s)
                write_summ(sfw, 'loss2/train', l2_train, s)
                write_summ(sfw, 'loss/dev', l_dev, s)
                write_summ(sfw, 'loss1/dev', l1_dev, s)
                write_summ(sfw, 'loss2/dev', l2_dev, s)
                write_summ(sfw, 'f1/train', f1_train, s)
                write_summ(sfw, 'f1/dev', f1_dev, s)
                write_summ(sfw, 'acc/train', acc_train, s)
                write_summ(sfw, 'acc/dev', acc_dev, s)
                sfw.flush()

                # save model
                save_file = os.path.join(save_dir, 'model_%d.ckpt' % s)
                saver.save(sess, save_file)


if __name__ == '__main__':
    main()
