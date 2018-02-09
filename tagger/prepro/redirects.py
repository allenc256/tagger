import argparse
import json
import re
import sys
from html.parser import HTMLParser

from tqdm import tqdm


class TagParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag = None
        self.attrs = {}
        self.data = None

    def handle_starttag(self, tag, attrs):
        self.tag = tag
        self.attrs = dict(attrs)

    def handle_data(self, data):
        self.data = data


def parse_tag(line):
    p = TagParser()
    p.feed(line.strip())
    return p


# parsing code adopted from https://github.com/attardi/wikiextractor

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*?>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4


def index_pages(lines):
    page_index = {}
    redirect_index = {}

    page_id = None
    ns = '0'
    last_id = None
    revid = None
    in_text = False
    redirect = None
    title = None

    for line in lines:
        if '<' not in line:  # faster than doing re.search()
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            redirect = None
        elif tag == 'id' and not page_id:
            page_id = m.group(3)
        elif tag == 'id' and page_id:
            revid = m.group(3)
        elif tag == 'title':
            title = parse_tag(line).data
        elif tag == 'ns':
            ns = parse_tag(line).data
        elif tag == 'redirect':
            redirect = parse_tag(line).attrs['title']
        elif tag == 'text':
            if m.lastindex == 3 and line[m.start(3)-2] == '/': # self closing
                # <text xml:space="preserve" />
                continue
            in_text = True
            line = line[m.start(3):m.end(3)]
            if m.lastindex == 4:  # open-close
                in_text = False
        elif tag == '/text':
            in_text = False
        elif in_text:
            pass
        elif tag == '/page':
            if page_id != last_id:
                page_index[page_id] = title
                if redirect:
                    redirect_index[title] = redirect
                last_id = page_id
                ns = '0'
            page_id = None
            revid = None
            title = None
            redirect = None

    return page_index, redirect_index


def resolve_redirects(redirect_index):
    r = {}
    v = set()

    def _resolve(t):
        if t in r:
            return r[t]
        if t in v:
            raise ValueError('infinite redirect')
        v.add(t)
        _t = redirect_index.get(t)
        _t = _resolve(_t) if _t else t
        r[t] = _t
        return _t

    for t in redirect_index.keys():
        _resolve(t)

    r = { k: v for k, v in r.items() if k != v }

    return r


def build_json_index(page_index, redirect_index):
    r = { title: { 'id': pid, 'from': [], 'to': None }
          for pid, title in page_index.items() }

    for t0, t1 in redirect_index.items():
        p0 = r.get(t0)
        p1 = r.get(t1)
        if p1:
            p1['from'].append(t0)
        else:
            print('warning - failed to resolve: %s (ignoring)' % t1, file=sys.stderr)
        if p0:
            p0['to'] = t1
        else:
            print('warning - failed to resolve: %s (ignoring)' % t0, file=sys.stderr)

    for p in r.values():
        if len(p['from']) <= 0:
            del p['from']
        if not p['to']:
            del p['to']

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='XML wiki dump file')
    parser.add_argument('output', help='output redirect index')

    args = parser.parse_args()

    with open(args.input, 'rt') as f:
        page_index, redirect_index = index_pages(tqdm(f))

    redirect_index = resolve_redirects(redirect_index)

    r = build_json_index(page_index, redirect_index)

    with open(args.output, 'wt') as f:
        json.dump(r, f, indent=1)


if __name__ == '__main__':
    main()
