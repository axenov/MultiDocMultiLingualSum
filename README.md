**To do:**
- [ ] look in detail Wikinews
- [ ] search other multi-lingual source of summaries
- [ ] begin scraping

# Multi-Document Multi-Lingual Summarization

## Scraping

Infos relative to scraping of Wikinews

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
