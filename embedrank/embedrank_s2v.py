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
        for w, pos in jp.lcut(document):
            if any(p in pos for p in ['n', 'l', 'v']):
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


if __name__ == "__main__":
    model_path = '/opt/algo_nfs/kdd_luozhouyang/embedrank/model.sent2vec.100d.bin.bin'
    model = Sent2VecEmbedRank(model_path)

    docs = [
        "java初级工程师(福田区)",
        "熟悉java开发，熟悉分布式，熟悉前端的react、vue框架。",
        "技能要求:1、计算机相关专业,本科及以上学历,2年以上java开发经验; 2、精通java,熟悉j2ee开发,熟练使用oracle/mssqlserver2005等数据库; 3、熟悉struts、spring、hiberate等框架; 4、熟悉常用的前端框架:ajax技术,熟悉jquery、extjs更佳; 5、有过金融/物流方面经验者优先。",
        "职责描述: 1.负责项目部分模块的设计开发工作; 2.指导协助同事解决日常工作中的问题 3.对系统存在的问题进行跟踪和定位并及时解决; 4.严格执行工作计划,主动汇报并高效完成任务保证部门及个人工作目标实现; 5.注重学习和积累,追求卓越; 任职要求: 1.计算机及相关专业本科,4年及以上工作经验; 2. 熟练使用java 编程,有良好的编码习惯;  •3.熟悉大数据处理相关产品架构和技术(如storm/hadoop/hive/hbase/spark/kafka/flume/zookeeper/redis等); 4.使用过storm/spark streaming优先; 5. 熟练使用linux开发环境和命令; 6.熟悉主流的数据库技术(oracle、mysql等); 7.具备良好的学习能力,分析解决问题和沟通表达能力",
    ]
    for d in docs:
        print(model.extract_keyword(d))
        print('=' * 100)
