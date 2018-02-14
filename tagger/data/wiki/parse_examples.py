import argparse
import json
import logging
import spacy as nlp

from tqdm import tqdm
from intervaltree import Interval, IntervalTree


def parse_examples(page_json, valid_targets, window_width):
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
        i0 = min(t.i for t in token_spans[l['start']:l['end']])
        i1 = max(t.i for t in token_spans[l['start']:l['end']])
        mid_point = (i0 + i1) // 2
        window_l = mid_point - (window_width // 2)
        window_r = window_l + window_width

        # create empty example
        example = []

        # build example
        for i in range(window_l, window_r):
            # if out of range, pad
            if i < 0 or i >= len(tokens):
                example.append(None)
                continue

            # find overlapping link(s)
            start = tokens[i].idx
            end = start + len(tokens[i])
            lis = link_spans[start:end]

            # extract link target
            if len(lis) == 1:
                target = next(iter(lis)).data['target']
            else:
                if len(lis) > 1:
                    logging.warning(
                        'link collision (ignoring): %s',
                        tokens[i])
                target = None

            # append token to example
            example.append({
                'text': tokens[i].text,
                'ent_type': tokens[i].ent_type_,
                'lemma': tokens[i].lemma_,
                'is_title': tokens[i].is_title,
                'like_num': tokens[i].like_num,
                'pos': tokens[i].pos_,
                'tag': tokens[i].tag_,
                'target': target
            })

        yield example


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input', default='parse_extracted.json',
        help='path to parsed wiki data')
    parser.add_argument(
        '-c', '--counter', default='parse_extracted.counter.json',
        help='path to link counter file')
    parser.add_argument(
        '-o', '--output', default='parse_examples.json',
        help='path to output examples file')
    parser.add_argument(
        '-l', '--log', default='parse_examples.log',
        help='processing log file')
    parser.add_argument(
        '-m', '--min', type=int, default=20,
        help='minimum link count for tags')
    parser.add_argument(
        '-w', '--window', type=int, default=120,
        help='example window width, in tokens')

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(
        filename=args.log, level=logging.INFO, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # determine valid targets
    with open(args.counter, 'rt') as f:
        link_counter = json.load(f)
        valid_targets = set(
            title for title, count in link_counter.items() if count >= args.min)
        logging.info(
            'link targets w/ %d min appearances: %d', args.min,
            len(valid_targets))

    # parse examples
    with open(args.output, 'wt') as out_file:
        with open(args.input, 'rt') as in_file:
            for line in tqdm(in_file):
                # parse JSON
                page = json.loads(line)
                logging.info('parsing page: %s', page['title'])

                # parse examples
                for e in parse_examples(page, valid_targets, args.window):
                    print(e)

                break # TEMPORARY


if __name__ == '__main__':
    main()
