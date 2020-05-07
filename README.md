# Multi-Document Multi-Lingual Summarization

## Prepare dataset

Tested only for german Wikinews for the moment.

### Download Wikinews dump

To download the german Wikinews dump run:

```
cd dumps/
wget https://dumps.wikimedia.org/dewikinews/latest/dewikinews-latest-pages-meta-current.xml.bz2
cd ../
```

### Create ``data.jsonl``

``data.jsonl`` contains one article per line stored in json format as follows:

```
{"title": title of the article, "text": list of paragraphs, "categories": list of categories, "sources": list of sources}
```

To create ``data.jsonl`` run:

```
cd data/
python data/create_data.py --wiki_dump_path WIKI_DUMP_PATH --max_doc_count MAX_DOC_COUNT --data_path DATA_PATH
cd ../
```

> Remarks: \
> The sources extraction is not perfect : links to an another Wikinews article are not taken, the domain url is sometimes present.\
> The text cleaning is not perfect : the main function to clean the text is ``filter_wiki`` and I noticed few bad cleaning. Run ``python data/failures.py`` for see one example.

### Stats

|      |    num_words |  num_sources |
| ---- | ------------ | ------------ |
|count | 13454 | 13454 |
|mean  |   220.4 |     2.8 |
|std   |   179.1 |     2.0 |
|min   |    11 |     1 |
|25%   |   114 |     1 |
|50%   |   174 |     2 |
|75%   |   269 |     4 |
|max   |  2713 |    25 |

To reproduce run: ``python data/stats.py``

### Explore dataset 

To see example of the dataset run: ``python data/display.py``

## Related Work

### English Multi Document Summarization

* [Multi-News: a Large-Scale Multi-Document SummarizationDataset and Abstractive Hierarchical Model](https://arxiv.org/pdf/1906.01749.pdf) | [github](https://github.com/Alex-Fabbri/Multi-News)\
Multi-news dataset from [newser](https://www.newser.com/) available.
* [Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf) | [github](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum)\
Great only decoder architecture with memory improvements to allow for input tokens but train a model from scratch would need to much data.
* [Hierarchical Transformers for Multi-Document Summarization](https://arxiv.org/pdf/1905.13164.pdf) | [github](https://github.com/nlpyang/hiersumm)\
WikiSum dataset avaible.

### Multi-Lingual Summarization

* [Sequential Transfer Learning in NLP forGerman Text Summarization](http://ceur-ws.org/Vol-2458/paper8.pdf)\
Use BERT as encoder.
* [Abstract Text Summarization: A Low Resource Challenge](https://www.aclweb.org/anthology/D19-1616.pdf)\
Data augmentation for summarization. Use OpenNMT-py transformers.

### English Abstract Summarization

* [BART: Denoising Sequence-to-Sequence Pre-training for NaturalLanguage Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) | [github](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)\
Great results on CNN-DM.
* [Exploring the Limits of Transfer Learning with aUnified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) | [github](https://github.com/google-research/text-to-text-transfer-transformer)\
Great results on CNN-DM.
* [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) | [github](https://github.com/nlpyang/PreSumm)\
Use BERT as encoder and a non-trained decoder. Two optimizers to avoid overfitting of BERT.

### Evaluation

* [EASY-M: Evaluation System for Multilingual Summarizers](https://www.aclweb.org/anthology/W19-89.pdf#page=63) | [web](https://summaryevaluation.azurewebsites.net/home)\
Online system that evaluate summaries with many metrics. Can be use to compare with our metrics.
