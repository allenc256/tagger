import argparse
import glob
import json
import logging
import re
import urllib.parse
from collections import Counter
from html.parser import HTMLParser

from tqdm import tqdm

from . import parse_redirects


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._fragments = []
        self._offset = 0
        self._curr_link_start = None
        self._curr_link_target = None

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        if self._curr_link_start is not None or self._curr_link_target:
            logging.warning('unmatched <a> tag detected (ignoring): %s', self._curr_link_target)
        self._curr_link_start = self._offset
        self._curr_link_target = urllib.parse.unquote(dict(attrs)['href'])

    def handle_endtag(self, tag):
        if tag != 'a':
            return
        if self._curr_link_start is None or self._curr_link_target is None:
            logging.warning('unmatched </a> tag detected (ignoring)')
        else:
            self.links.append({
                'target': self._curr_link_target,
                'start': self._curr_link_start,
                'end': self._offset
            })
        self._curr_link_start = None
        self._curr_link_target = None

    def handle_data(self, data):
        self._fragments.append(data)
        self._offset += len(data)

    def get_sanitized_text(self):
        return ''.join(self._fragments)


def normalize_title(title):
    title = title.strip(' _')
    title = re.sub(r'[\s_]+', ' ', title)
    if len(title) == 0:
        return title
    # capitalize first char
    return title[0].upper() + title[1:]


def parse_page(page_json):
    p = TextParser()
    p.feed(page_json['text'])
    page_json['text'] = p.get_sanitized_text()
    page_json['links'] = p.links
    return page_json


def sanitize_links(page, redirect_index):
    result = []

    for l in page['links']:
        # normalize target
        l['target'] = normalize_title(l['target'])

        # resolve redirect
        t = redirect_index.get(l['target'])
        if not t:
            logging.warning('unresolvable target (ignoring): %s', l['target'])
            continue
        l['target'] = t

        # append
        result.append(l)

    # replace links
    page['links'] = result

    return page


def count_links(page, link_counter):
    for l in page['links']:
        link_counter[l['target']] += 1


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='path to original XML dump file')
    parser.add_argument('extracted', help='path to extraction directory')
    parser.add_argument(
        '-o', '--output', default='parse_links.json',
        help='path to parsing output file')
    parser.add_argument(
        '-l', '--log', default='parse_links.log', help='processing log file')
    parser.add_argument(
        '-c', '--counter', default='parse_links.counter.json',
        help='path to link counter file')

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(
        filename=args.log, level=logging.INFO, filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # parse redirect index
    with open(args.input, 'rt') as f:
        redirect_index = parse_redirects(f)

    # link counter
    link_counter = Counter()

    # parse extracted file
    with open(args.output, 'wt') as out_file:
        in_file_names = glob.glob('%s/**/wiki_*' % args.input, recursive=True)
        for in_file_name in tqdm(in_file_names, desc='parsing links'):
            logging.info('processing file: %s', in_file_name)
            # parse each file (contains multiple pages)
            with open(in_file_name, 'rt') as in_file:
                # parse each line (one line = one page)
                for l in in_file:
                    # parse JSON
                    page = json.loads(l)
                    logging.info('parsing page: %s', page['title'])

                    # parse page
                    page = parse_page(page)

                    # fix links / redirects
                    page = sanitize_links(page, redirect_index)

                    # count links
                    count_links(page, link_counter)

                    # output (w/ extra newline separator)
                    json.dump(page, out_file)
                    print(file=out_file)

    # write link counter
    with open(args.counter, 'wt') as f:
        print('{', file=f)
        for title, count in link_counter.most_common():
            print(' %s: %d,' % (json.dumps(title), count), file=f)
        print('}', file=f)


if __name__ == '__main__':
    main()
