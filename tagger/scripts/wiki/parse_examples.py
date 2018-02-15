import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import queue
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


def feed_process(in_queue, filename, num_pages):
    # feed input queue
    with open(filename, 'rt') as f:
        lines = itertools.islice(f, 0, num_pages)
        for l in tqdm(lines, desc='parsing examples', total=num_pages):
            in_queue.put(l)

    # wait for all items to complete processing; this ensures the parent
    # process knows when things are actually done
    in_queue.join()


def parse_process(in_queue, out_queue, nlp, valid_targets, window_width):
    while True:
        # parse next page
        line = in_queue.get()
        exs = parse_page(line, nlp, valid_targets, window_width)

        # write parsed examples to queue
        out_queue.put(exs)

        # mark task done
        in_queue.task_done()


def parse_page(line, nlp, valid_targets, window_width):
    # parse JSON
    page_json = json.loads(line)

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
    examples = []
    for l in valid_links:
        overlaps = [ival.data.i for ival in token_spans[l['start']:l['end']]]

        # ignore empty links
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

        # fill example with features
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
                        'link collision (ignoring): %s %s',
                        tokens[i], [ival.data for ival in ivals])
                target = None

            # append features
            example.text.append(sanitize_text(tokens[i].text))
            example.ent_type.append(tokens[i].ent_type_)
            example.is_title.append(str(tokens[i].is_title))
            example.like_num.append(str(tokens[i].like_num))
            example.pos.append(tokens[i].pos_)
            example.tag.append(tokens[i].tag_)
            example.target.append(target)

        examples.append(example)

    # logging
    logging.info(
        'parsed %d examples for page: %s', len(examples), page_json['title'])

    return examples


def count_links(file):
    c = Counter()
    num_pages = 0
    for line in tqdm(file, desc='counting links'):
        page = json.loads(line)
        num_pages += 1
        for l in page['links']:
            c[l['target']] += 1
    return c, num_pages


def dump_example(ex):
    logging.debug('dumping example:')
    for i in range(len(ex.text)):
        args = [getattr(ex, fn)[i] for fn in [
            'text', 'target', 'ent_type', 'is_title', 'like_num', 'pos', 'tag']]
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


def parse_examples(args, nlp, valid_targets, num_pages):
    root_logger = logging.getLogger()
    buffer = []

    # queues
    in_queue = mp.JoinableQueue(args.processes * 2)
    out_queue = mp.JoinableQueue(args.processes * 2)

    # start feed process
    feeder = mp.Process(
        target=feed_process, args=(in_queue, args.input, num_pages))
    feeder.start()

    # start parser proceses
    for i in range(args.processes):
        p = mp.Process(
            target=parse_process,
            args=(in_queue, out_queue, nlp, valid_targets, args.window))
        p.daemon = True
        p.start()

    # read results
    while feeder.is_alive():
        try:
            es = out_queue.get(timeout=1)
        except queue.Empty:
            continue

        # dump examples
        if root_logger.isEnabledFor(logging.DEBUG):
            for e in es:
                dump_example(e)

        # output examples for page
        if len(buffer) >= args.buffer_size > 0:
            yield remove_random(buffer)
        buffer.append(es)

    # yield remaining elements in buffer
    random.shuffle(buffer)
    for es in buffer:
        yield es


def convert_to_tfrecord(ex, data_dict):
    fs = {}
    for name in data_dict.feature_names():
        fdict = getattr(data_dict, name)
        fvals = getattr(ex, name)
        fids = [fdict.get_id(v) for v in fvals]
        fs[name] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=fids))
    return tf.train.Example(features=tf.train.Features(feature=fs))


def write_examples(args, examples, data_dict):
    train_page_count = 0
    train_ex_count = 0
    dev_page_count = 0
    dev_ex_count = 0
    opts = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(args.train, options=opts) as train_writer:
        with tf.python_io.TFRecordWriter(args.dev, options=opts) as dev_writer:
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
                    r = convert_to_tfrecord(e, data_dict)
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
        default='data/wiki/parsed_examples.train.tfrecords.gz',
        help='path to output training set file')
    parser.add_argument(
        '--dev', metavar='FILE',
        default='data/wiki/parsed_examples.dev.tfrecords.gz',
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
        '--buffer-size', metavar='N', type=int, default=1000,
        help='maximum size of shuffle buffer (-1 to load everything into '
             'memory)')
    parser.add_argument(
        '--processes', metavar='N', type=int,
        default=max(mp.cpu_count() // 2, 1),
        help='number of threads to perform parsing on')
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
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')

    # determine valid targets
    with open(args.input, 'rt') as f:
        link_counter, num_pages = count_links(f)
        valid_targets = set(
            title for title, count in link_counter.items() if count >= args.min)
        logging.info(
            'link targets w/ %d min appearances: %d/%d', args.min,
            len(valid_targets), len(link_counter.keys()))

    # apply limit
    if args.limit > 0:
        num_pages = min(args.limit, num_pages)

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
    examples = parse_examples(args, nlp, valid_targets, num_pages)

    # write examples
    write_examples(args, examples, data_dict)

    # write data dictionary
    with open(args.dict, 'wt') as f:
        json.dump(data_dict.to_json(), f, indent=1)

if __name__ == '__main__':
    main()
