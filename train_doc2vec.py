import argparse
import logging
import os
import re

import gensim
import jieba
import jieba.posseg as jp
import smart_open

INVALID_TOKEN_PATTERN = re.compile(r'^\W+$')
NUMBERS_PATTERN = re.compile(r'[0-9]+.?')


def tokenize(text):
    tokens = []
    for w, pos in jp.lcut(text.lower()):
        if not pos:
            continue
        if INVALID_TOKEN_PATTERN.match(w):
            continue
#         if NUMBERS_PATTERN.match(w):
#             continue
        if not w.strip():
            continue
        tokens.append(w.strip())
    return tokens


def read_corpus(fname):
    with smart_open.open(fname, encoding="utf8") as f:
        for i, line in enumerate(f):
            tokens = tokenize(line.strip('\n').strip())
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def train(input_file, model_path, vocab_path, epochs=10, **kwargs):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=7, max_vocab_size=100000, works=16)
    train_corpus = read_corpus(input_file)
    model.build_vocab(train_corpus)

    vocab_dir = '/'.join(str(vocab_path).split('/')[:-1])
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    with open(vocab_path, mode='wt', encoding='utf8') as fout:
        for k, v in model.wv.vocab.items():
            fout.write(k + '\t' + str(v.count) + '\n')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=10)
    model.save(model_path)

    # for epoch in range(epochs):
    #     logging.info('Start training in epoch: %d' % epoch)
    #     model.train(train_corpus, total_examples=model.corpus_count, epochs=1)
    #     logging.info('Finished epoch %d' % epoch)
    #     output = model_path + '.ckpt.' + str(epoch)
    #     model.save(output)
    #     logging.info('Saved ckpt to %s' % output)

    logging.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--model_save_path')
    parser.add_argument('--vocab_save_path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--jieba_user_dict', default=None)

    args, _ = parser.parse_known_args()

    logging.basicConfig(filename='log/train.doc2vec.log', level=logging.INFO)

    if args.jieba_user_dict:
        jieba.load_userdict(args.jieba_user_dict)
        logging.info('Load userdict finished, path is %s' % args.jieba_user_dict)
    jieba.initialize()

    train(args.input_file, args.model_save_path, args.vocab_save_path, args.epochs)
