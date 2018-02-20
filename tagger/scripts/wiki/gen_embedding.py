import argparse
import json

import fastText
import numpy as np

from tagger.model.dataset import DatasetDictionary


def main():
    # parse command-line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dict', default='data/wiki/parsed_examples.dict.json',
        help='dataset/features dictionary file')
    parser.add_argument(
        '--model', default='data/embeddings/wiki.en.bin',
        help='fastText binary model file')
    parser.add_argument(
        '--output', default='data/wiki/parsed_examples.embedding.npy',
        help='output embedding file')
    args = parser.parse_args()

    # load feature dict
    with open(args.dict, 'rt') as f:
        data_dict = DatasetDictionary()
        data_dict.from_json(json.load(f))

    # load model
    print('loading fastText model...', end='', flush=True)
    model = fastText.FastText.load_model(args.model)
    print(' done')

    # build embedding matrix
    embedding = np.zeros([data_dict.text.size(), model.get_dimension()])
    out_count = 0
    for i in range(data_dict.text.size()):
        word = data_dict.text.value(i)
        if word:
            if model.get_word_id(word) < 0:
                out_count += 1
            embedding[i, :] = model.get_word_vector(word)

    # save embedding matrix
    with open(args.output, 'wb') as f:
        np.save(f, embedding)

    print('computed %d vectors (%d out of vocab)'
          % (data_dict.text.size(), out_count))


if __name__ == '__main__':
    main()
