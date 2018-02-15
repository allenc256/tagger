import argparse
import json
import logging
import os
import random
from collections import Counter
from types import SimpleNamespace

import spacy
import tensorflow as tf
from intervaltree import IntervalTree
from tqdm import tqdm

from tagger.model.feature import DatasetDictionary


def sanitize_text(text):
    text = text.strip()
    text = text.lower()
    return text


def parse_examples_page(nlp, page_json, valid_targets, window_width, data_dict):
    # tokenize
    tokens = next(nlp.pipe([page_json['text']]))

    # index tokens
    token_spans = IntervalTree()
    for i, token in enumerate(tokens):
        start = token.idx
        end = start + len(token)
        # N.B., will throw exn on zero-length spans
        if start < end:
            token_spans[start:end] = token
        assert token.i == i

    # filter to valid links
    valid_links = [
        l for l in page_json['links']
        if l['target'] in valid_targets and l['start'] < l['end']]

    # index links
    link_spans = IntervalTree()
    for l in valid_links:
        link_spans[l['start']:l['end']] = l

    # create examples (one per valid link)
    for l in valid_links:
        overlaps = [ival.data.i for ival in token_spans[l['start']:l['end']]]
        # empty link
        if len(overlaps) == 0:
            logging.warning('found non-overlapping link (ignoring): %s', l)
            continue

        # find window endpoints
        mid_point = (min(overlaps) + max(overlaps)) // 2
        window_l = max(0, mid_point - (window_width // 2))
        window_r = min(len(tokens), window_l + window_width)

        # create empty example
        example = SimpleNamespace(
            text=[], ent_type=[], is_title=[], like_num=[], pos=[], tag=[],
            target=[])

        # build example
        for i in range(window_l, window_r):
            # find overlapping link(s)
            start = tokens[i].idx
            end = start + len(tokens[i])
            ivals = link_spans[start:end]

            # extract link target
            if len(ivals) == 1:
                target = next(iter(ivals)).data['target']
            else:
                if len(ivals) > 1:
                    logging.warning(
                        'link collision (ignoring): %s',
                        tokens[i])
                target = None

            # append to example
            example.text.append(
                data_dict.text.get_id(sanitize_text(tokens[i].text)))
            example.ent_type.append(
                data_dict.ent_type.get_id(tokens[i].ent_type_))
            example.is_title.append(
                data_dict.is_title.get_id(str(tokens[i].is_title)))
            example.like_num.append(
                data_dict.like_num.get_id(str(tokens[i].like_num)))
            example.pos.append(
                data_dict.pos.get_id(tokens[i].pos_))
            example.tag.append(
                data_dict.tag.get_id(tokens[i].tag_))
            example.target.append(
                data_dict.target.get_id(target))

        yield example


def count_links(file):
    c = Counter()
    for line in tqdm(file, desc='counting links'):
        page = json.loads(line)
        for l in page['links']:
            c[l['target']] += 1
    return c


def dump_example(ex, data_dict):
    logging.debug('dumping example:')
    for i in range(len(ex.text)):
        args = [getattr(data_dict, fn).get_value(getattr(ex, fn)[i])
                for fn in ['text', 'target', 'ent_type', 'is_title',
                           'like_num', 'pos', 'tag']]
        logging.debug('%20.20s %20.20s %10.10s %5.5s %5.5s %5.5s %5.5s', *args)


def remove_random(buffer):
    """
    Removes an element from the buffer at random and returns it.
    May reorder elements in the buffer.
    :param buffer: buffer to remove random element from
    :return: randomly removed element
    """
    i = random.randrange(0, len(buffer))
    v = buffer[i]
    buffer[i] = buffer[-1]
    buffer.pop()
    return v


def parse_examples_all(args, nlp, data_dict, valid_targets):
    root_logger = logging.getLogger()
    buffer = []
    page_count = 0

    # parse examples
    with open(args.input, 'rt') as in_file:
        for line in tqdm(in_file, desc='parsing examples'):
            # parse JSON
            page = json.loads(line)

            # parse examples
            es = list(parse_examples_page(
                nlp, page, valid_targets, args.window, data_dict))

            # logging
            logging.info(
                'parsed %d examples for page: %s', len(es), page['title'])

            # skip pages without examples
            if len(es) <= 0:
                continue

            # check limit
            page_count += 1
            if 0 <= args.limit < page_count:
                break

            # dump examples
            if root_logger.isEnabledFor(logging.DEBUG):
                for e in es:
                    dump_example(e, data_dict)

            # output examples for page
            if len(buffer) >= args.buffer_size > 0:
                yield remove_random(buffer)
            buffer.append(es)

    # yield remaining elements in buffer
    random.shuffle(buffer)
    for es in buffer:
        yield es


def convert_to_tfrecord(ex, feature_names):
    return tf.train.Example(features=tf.train.Features(
        feature={fn:tf.train.Feature(
            int64_list=tf.train.Int64List(value=getattr(ex, fn)))
            for fn in feature_names}))


def write_examples(args, examples, data_dict):
    train_page_count = 0
    train_ex_count = 0
    dev_page_count = 0
    dev_ex_count = 0

    with tf.python_io.TFRecordWriter(args.train) as train_writer:
        with tf.python_io.TFRecordWriter(args.dev) as dev_writer:
            for es in examples:
                # randomly pick which set to add example to
                if random.random() <= args.dev_fraction:
                    writer = dev_writer
                    dev_page_count += 1
                    dev_ex_count += len(es)
                else:
                    writer = train_writer
                    train_page_count += 1
                    train_ex_count += len(es)

                # write examples
                for e in es:
                    r = convert_to_tfrecord(e, data_dict.feature_names())
                    writer.write(r.SerializeToString())

    logging.info(
        'wrote %d examples (from %d pages) to train set: %s',
        train_ex_count, train_page_count, args.train)
    logging.info(
        'wrote %d examples (from %d pages) to dev set: %s',
        dev_ex_count, dev_page_count, args.dev)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', metavar='FILE', default='data/wiki/parsed_wiki.json',
        help='path to parsed wiki data')
    parser.add_argument(
        '--train', metavar='FILE',
        default='data/wiki/parsed_examples.train.tfrecords',
        help='path to output training set file')
    parser.add_argument(
        '--dev', metavar='FILE',
        default='data/wiki/parsed_examples.dev.tfrecords',
        help='path to output development set file')
    parser.add_argument(
        '--dev-fraction', metavar='F', type=float, default=0.02,
        help='approximate fraction of examples to write to development set')
    parser.add_argument(
        '--dict', metavar='FILE',
        default='data/wiki/parsed_examples.dict.json',
        help='path to output dataset/features dictionary file')
    parser.add_argument(
        '--fast', action='store_true',
        help='disable more expensive NLP featurization')
    parser.add_argument(
        '--log', metavar='FILE', default='logs/parse_examples.log',
        help='processing log file')
    parser.add_argument(
        '--min', metavar='N', type=int, default=20,
        help='minimum link count for tags')
    parser.add_argument(
        '--window', metavar='N', type=int, default=120,
        help='example window width, in tokens')
    parser.add_argument(
        '--debug', action='store_true', help='enable detailed debug logging')
    parser.add_argument(
        '--limit', metavar='N', type=int, default=-1,
        help='maximum number of pages to parse (-1 for no limit)')
    parser.add_argument(
        '--buffer-size', metavar='N', type=int, default=10000,
        help='maximum size of shuffle buffer (-1 to load everything into '
             'memory)')
    args = parser.parse_args()

    # make sure directories exist
    os.makedirs(os.path.dirname(args.dev), exist_ok=True)
    os.makedirs(os.path.dirname(args.train), exist_ok=True)
    os.makedirs(os.path.dirname(args.dict), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    # configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=args.log, level=level, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # determine valid targets
    with open(args.input, 'rt') as f:
        link_counter = count_links(f)
        valid_targets = set(
            title for title, count in link_counter.items() if count >= args.min)
        logging.info(
            'link targets w/ %d min appearances: %d/%d', args.min,
            len(valid_targets), len(link_counter.keys()))

    # load spacy
    print('loading spacy...', end='', flush=True)
    nlp = spacy.load('en')
    print(' done')

    # disable pipes
    if args.fast:
        logging.info('fast mode enabled --- disabling expensive NLP pipes')
        nlp.disable_pipes('tagger', 'ner', 'parser')

    # init data dictionaries
    data_dict = DatasetDictionary()

    # parse examples
    examples = parse_examples_all(args, nlp, data_dict, valid_targets)

    # write examples
    write_examples(args, examples, data_dict)

    # write data dictionary
    with open(args.dict, 'wt') as f:
        json.dump(data_dict.to_json(), f, indent=1)

if __name__ == '__main__':
    main()
