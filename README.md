# EmbedRank

This code is refactored from [yagays/embedrank](https://github.com/yagays/embedrank)

Paper: 

* [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://arxiv.org/abs/1801.04470)

## Usage

Here are two kinds of `EmbedRank`:

* `Sent2VecEmbedRank`, using `sent2vec` model to get phrase embedding and sentence embedding
* `Doc2VecEmbedRank`, using `doc2vec` model to get phrase embedding and sentence embedding


For `Sent2VecEmbedRank` model:

```python
from embedrank import Sent2VecEmbedRank

model_path = '/path/to/your/pretrained/sent2vec/model'

model = Sent2VecEmbedRank(model_path)

docs = [
    "java初级工程师(福田区)",
    "熟悉java开发，熟悉分布式，熟悉前端的react、vue框架。",
]

for doc in docs:
  print(model.extract_keyword(doc))
  print('=' * 100)
```

```bash
[('java', 0.48793846), ('福田区', 0.439785), ('工程师', 0.11300646)]
====================================================================================================
[('react', 0.56610876), ('熟悉', 0.37888753), ('java', 0.42108417), ('框架', 0.3385066), ('分布式', 0.2882378), ('开发', 0.22691125), ('vue', 0.45964164), ('熟悉', 0.37888753), ('熟悉', 0.37888753)]
====================================================================================================
```

For `Doc2VecEmbedRank` model:

```python
from embedrank import Doc2VecEmbedRank

model_path = '/path/to/your/pretrained/doc2vec/model'

model = Doc2VecEmbedRank(model_path)

docs = [
    "java初级工程师(福田区)",
    "熟悉java开发，熟悉分布式，熟悉前端的react、vue框架。",
]

for doc in docs:
  print(model.extract_keyword(doc))
  print('=' * 100)

```

```bash
[('java', 0.6828749), ('工程师', 0.63509357), ('福田区', 0.52648664)]
====================================================================================================
[('分布式', 0.61760384), ('java', 0.61595225), ('熟悉', 0.43337262), ('react', 0.3199829), ('开发', 0.34032723), ('框架', 0.28698522), ('vue', 0.25441816), ('熟悉', 0.44235963), ('熟悉', 0.4399612)]
====================================================================================================
```

## Pretrain models

It's very easy to pretrain either a `sent2vec` or `doc2vec` model.

Train `sent2vec` model:

* [train-sent2vec-model](https://github.com/epfml/sent2vec#train-a-new-sent2vec-model)

Train `doc2vec` model:

* [train-doc2vec-model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)

