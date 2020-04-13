import logging
import re

import jieba
import jieba.posseg as jp
import numpy as np
import sent2vec
from sklearn.metrics.pairwise import cosine_similarity

# TODO(zhouyang.luo) jieba load user dict


class Sent2VecEmbedRank(object):

    def __init__(self, model_path):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(model_path, inference_mode=True)

    def extract_keyword(self, document, _lambda=0.5, N=20):
        phrase_ids, phrases, similarities = self._mmr(document, _lambda, N)
        if len(phrases) == 0:
            logging.warning('No phrases collected from document: %s' % document)
            return []
        if len(phrase_ids) == 0:
            logging.warning('No phrases collected from document: %s' % document)
            return []
        if len(similarities) == 0:
            logging.warning('Phrase-Document similarity matrix is empty.')
            return []

        outputs = []
        for idx in phrase_ids:
            outputs.append((phrases[idx], similarities[idx][0]))
        return outputs

    def _mmr(self, document, _lambda=0.5, N=20):
        tokens = self._tokenize(document)
        if len(tokens) == 0:
            logging.warning('No tokens return by tokenization. document: %s' % document)
            return [], [], []
        document_embedding = self.model.embed_sentence(' '.join(tokens))
        phrases, phrase_embeddings = self._create_phrases_with_embeddings(document)
        if len(phrases) == 0:
            return [], [], []
        phrase_embeddings = np.array(phrase_embeddings)  # shape (num_phrases, embedding_size)
        N = min(N, len(phrases))
        # similarity between each phrase and document
        phrase_document_similarities = cosine_similarity(phrase_embeddings, document_embedding)
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
        # TODO(zhouyang.luo) fix phrase generation
        phrases = []
        embeddings = []
        for w, pos in jp.cut(document):
            if any(p in pos for p in ['n', 'l', 'v', 'i']):
                phrases.append(w)
                embeds = self.model.embed_unigrams([w])
                embeddings.append(embeds[0])
                # print({'word': w, 'embedding': embeds[0]})
        return phrases, embeddings

    def _tokenize(self, document):
        tokens = []
        for w in jieba.cut(document):
            w = w.strip()
            if not w:
                continue
            tokens.append(w)
        return tokens
