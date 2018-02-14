import argparse
import json
import logging
import random
import tensorflow as tf
from collections import Counter
from types import SimpleNamespace

import spacy
from intervaltree import IntervalTree
from tqdm import tqdm


class FeatureDictionary:
    def __init__(self):
        # ID zero always reserved
        self.id_to_value = [None]
        self._value_to_id = {None: 0}

    def get_id(self, value):
        fid = self._value_to_id.get(value)
        if fid is None:
            fid = len(self.id_to_value)
            self._value_to_id[value] = fid
            self.id_to_value.append(value)
        return fid

    def get_value(self, fid):
        return self.id_to_value[fid]


def sanitize_text(text):
    text = text.strip()
    text = text.lower()
    return text


def parse_examples_page(nlp, page_json, valid_targets, window_width, feature_dicts):
    # tokenize
    tokens = nlp(page_json['text'])

    # index tokens
    token_spans = IntervalTree()
    for i, token in enumerate(tokens):
        start = token.idx
        end = start + len(token)
        token_spans[start:end] = token
        assert token.i == i

    # filter to valid links
    valid_links = [
        l for l in page_json['links'] if l['target'] in valid_targets]

    # index links
    link_spans = IntervalTree()
    for l in valid_links:
        link_spans[l['start']:l['end']] = l

    # create examples (one per valid link)
    for l in valid_links:
        # find window endpoints
        i0 = min(ival.data.i for ival in token_spans[l['start']:l['end']])
        i1 = max(ival.data.i for ival in token_spans[l['start']:l['end']])
        mid_point = (i0 + i1) // 2
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
                feature_dicts.text.get_id(sanitize_text(tokens[i].text)))
            example.ent_type.append(
                feature_dicts.ent_type.get_id(tokens[i].ent_type_))
            example.is_title.append(
                feature_dicts.is_title.get_id(str(tokens[i].is_title)))
            example.like_num.append(
                feature_dicts.like_num.get_id(str(tokens[i].like_num)))
            example.pos.append(
                feature_dicts.pos.get_id(tokens[i].pos_))
            example.tag.append(
                feature_dicts.tag.get_id(tokens[i].tag_))
            example.target.append(
                feature_dicts.target.get_id(target))

        yield example


def count_links(file):
    c = Counter()
    for line in tqdm(file, desc='counting links'):
        page = json.loads(line)
        for l in page['links']:
            c[l['target']] += 1
    return c


FEATURE_KEYS = [
    'text', 'target', 'ent_type', 'is_title', 'like_num', 'pos', 'tag']


def dump_example(ex, feature_dicts):
    logging.debug('dumping example:')
    for i in range(len(ex.text)):
        args = [getattr(feature_dicts, key).get_value(getattr(ex, key)[i])
                for key in FEATURE_KEYS]
        logging.debug('%20.20s %20.20s %10.10s %5.5s %5.5s %5.5s %5.5s', *args)


def parse_examples_all(args, nlp, feature_dicts, valid_targets):
    # get root logger
    root_logger = logging.getLogger()

    # example buffer
    shuffle_buffer = []

    # parse examples
    with open(args.input, 'rt') as in_file:
        for line_no, line in tqdm(enumerate(in_file), desc='parsing examples'):
            # check limit
            if 0 <= args.debug < line_no + 1:
                break

            # parse JSON
            page = json.loads(line)
            logging.info('parsing page: %s', page['title'])

            # parse examples
            for e in parse_examples_page(nlp, page, valid_targets, args.window, feature_dicts):
                if root_logger.isEnabledFor(logging.DEBUG):
                    dump_example(e, feature_dicts)
                if args.shuffle:
                    shuffle_buffer.append(e)
                else:
                    yield e

    # optionally shuffle
    if args.shuffle:
        random.shuffle(shuffle_buffer)
        for e in shuffle_buffer:
            yield e


def convert_to_tfrecord(ex):
    return tf.train.Example(features=tf.train.Features(
        feature={key:tf.train.Feature(
            int64_list=tf.train.Int64List(value=getattr(ex, key)))
            for key in FEATURE_KEYS}))


def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input', default='parsed_wiki.json',
        help='path to parsed wiki data')
    parser.add_argument(
        '-o', '--output', default='parsed_examples.tfrecords',
        help='path to output examples file')
    parser.add_argument(
        '-f', '--feature-dicts', default='parsed_examples.feature_dicts.json',
        help='path to output feature dictionaries file')
    parser.add_argument(
        '--log', default='parse_examples.log',
        help='processing log file')
    parser.add_argument(
        '--min', type=int, default=20,
        help='minimum link count for tags')
    parser.add_argument(
        '--window', type=int, default=120,
        help='example window width, in tokens')
    parser.add_argument(
        '--debug', type=int, metavar='LIMIT', default=-1,
        help='enable detailed debug logging, and optionally only parse up to '
             'a maximum number of pages')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='shuffle examples before writing (requires loading all examples '
             'in-memory!)')
    args = parser.parse_args()

    # configure logging
    level = logging.DEBUG if args.debug >= 0 else logging.INFO
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
    nlp = spacy.load('en')

    # init feature dictionaries
    feature_dicts = SimpleNamespace()
    for key in FEATURE_KEYS:
        setattr(feature_dicts, key, FeatureDictionary())

    # parse examples
    examples = parse_examples_all(args, nlp, feature_dicts, valid_targets)

    # write examples
    with tf.python_io.TFRecordWriter(args.output) as w:
        for e in examples:
            w.write(convert_to_tfrecord(e).SerializeToString())

    # write feature dictionaries
    with open(args.feature_dicts, 'wt') as f:
        json.dump(
            {k:v.id_to_value for k, v in feature_dicts.__dict__.items()},
            f, indent=1)

if __name__ == '__main__':
    main()
