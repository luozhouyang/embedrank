import logging
import re

import jieba
import jieba.posseg as jp
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

INVALID_PATTERN = re.compile(r'^\W+$')

# TODO(zhouyang.luo) jieba load user dict


class Doc2VecEmbedRank(object):

    def __init__(self, model_path):
        self.model = Doc2Vec.load(model_path)

    def extract_keyword(self, document, _lambda=0.5, N=20):
        phrase_ids, phrases, similarities = self._mmr(document, _lambda, N)
        if len(phrases) == 0:
            logging.warn('No phrases collected from document.')
            return []
        if len(phrase_ids) == 0:
            logging.warn('No phrases collected from document.')
            return []
        if len(similarities) == 0:
            logging.warn('Phrase-Document similarity matrix is empty.')
            return []

        outputs = []
        for idx in phrase_ids:
            outputs.append((phrases[idx], similarities[idx][0]))
        return outputs

    def _mmr(self, document, _lambda=0.5, N=20):
        tokens = self._tokenize(document)
        if len(tokens) == 0:
            logging.warn('No tokens return by tokenization.')
            return [], [], []
        document_embedding = self.model.infer_vector(tokens)
        phrases, phrase_embeddings = self._create_phrases_with_embeddings(document)
        if len(phrases) == 0:
            return [], [], []
        phrase_embeddings = np.array(phrase_embeddings)  # shape (num_phrases, embedding_size)
        N = min(N, len(phrases))
        # similarity between each phrase and document
        phrase_document_similarities = cosine_similarity(phrase_embeddings, document_embedding.reshape(1, -1))
        # similarity between phrases
        phrase_phrase_similarities = cosine_similarity(phrase_embeddings)

        # MMR
        # 1st iteration
        unselected = list(range(len(phrases)))
        select_idx = np.argmax(phrase_document_similarities)  # most similiar phrase of document

        selected = [select_idx]
        unselected.remove(select_idx)

        # other iterations
        for _ in range(N - 1):
            mmr_distance_to_doc = phrase_document_similarities[unselected, :]
            mmr_distance_between_phrases = np.max(phrase_phrase_similarities[unselected][:, selected], axis=1)

            mmr = _lambda * mmr_distance_to_doc - (1 - _lambda) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected, phrases, phrase_document_similarities

    def _create_phrases_with_embeddings(self, document):
        # TODO(zhouyang.luo) fix phrase generation jieba.cutall()
        phrases = []
        embeddings = []
        for w, pos in jp.lcut(document):
            if any(p in pos for p in ['n', 'l', 'v']):
                phrases.append(w)
                vector = self.model.infer_vector([w])
                embeddings.append(vector)
        return phrases, embeddings

    def _tokenize(self, document):
        tokens = []
        for w in jieba.cut(document):
            w = w.strip()
            if not w:
                continue
            if INVALID_PATTERN.match(w):
                continue
            tokens.append(w)
        return tokens

