# Multi-Document Multi-Lingual Summarization

## Prepare dataset

### Clone repository 

```
git clone git clone https://github.com/airKlizz/MultiDocMultiLingualSum
cd MultiDocMultiLingualSum/
```

### Download Wikinews dump

To download the german Wikinews dump run:

```
cd dataset/wikinews/dumps/
wget https://dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-meta-current.xml.bz2
cd ../../..
```

### Create json files

Json files contain one article stored in json format as follows:

```json
{
  "title": title of the article, 
  "text": list of paragraphs, 
  "categories": ['categorie 1', 'categorie 2', ...] ,
  "sources": ['sources 1', 'sources 2', ...] 
}
```

To create the json files run:

```
python dataset/wikinews/create_data.py --wiki_dump_path 'dataset/wikinews/dumps/enwikinews-latest-pages-meta-current.xml.bz2' \
                                        --max_doc_count 0 \
                                        --index_path 'dataset/wikinews/index/en.wikinews.index' \
                                        --json_path 'dataset/wikinews/json.en/'
```

> Remarks: \
> The sources extraction is not perfect : links to an another Wikinews article are not taken.\
> The text cleaning is not perfect : the main function to clean the text is ``filter_wiki`` and I noticed few bad cleaning. Run ``python dataset/wikinews/failures.py`` for see one example.

|        | German | English | French |
| --- | --- | --- | --- |
|Pages read | 60427 | 2811419 | xxx |
|Pages returned | 13454 | 16616 | xxx |
|Wrong namespace | 46290 | 2769094 | xxx |
|No sources | 676 | 5742 | xxx |
|No text | 2 | 36 | xxx |
|Redirect | 5 | 19931 | xxx |

### Index sources

Extract html and content from source urls of Wikinews articles. The script uses the [archive](https://web.archive.org/) version of the page if it exists otherwise it archives the page.

```
python dataset/index_sources.py -wikinews_json_path 'dataset/wikinews/json.en' \
                                --index_path 'dataset/sources/index/en.sources.index' \
                                --html_path 'dataset/sources/html.de' \
                                --json_path 'dataset/sources/json.de' \
                                --max_url_count -1 \
                                --max_workers 10
```

### Stats

To reproduce run: ``python dataset/wikinews/stats.py --data_path DATA_PATH ``

#### English Wikinews

|      |    num_words |  num_sources |
| ---- | ------------ | ------------ |
|count | 16616 | 16616 |
|mean  |   304.3 |     3.0 |
|std   |   290.5 |     2.5 |
|min   |     1 |     1 |
|25%   |   170 |     2 |
|50%   |   242 |     3 |
|75%   |   355 |     4 |
|max   | 11629 |    52 |

## Related Work

### English Multi Document Summarization

* [Multi-News: a Large-Scale Multi-Document SummarizationDataset and Abstractive Hierarchical Model](https://arxiv.org/pdf/1906.01749.pdf) | [github](https://github.com/Alex-Fabbri/Multi-News)\
Multi-news dataset from [newser](https://www.newser.com/) available.
* [Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf) | [github](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum)\
Great only decoder architecture with memory improvements to allow for input tokens but train a model from scratch would need to much data.
* [Hierarchical Transformers for Multi-Document Summarization](https://arxiv.org/pdf/1905.13164.pdf) | [github](https://github.com/nlpyang/hiersumm)\
WikiSum dataset avaible.
* [Towards Automatic Construction of News Overview Articles by NewsSynthesis](https://www.aclweb.org/anthology/D17-1224.pdf)\
Use part of (top 100 articles in length) english Wikinews as test dataset and evaluate several methods but not deep learning methods.
* [A Zipf-Like Distant Supervision Approachfor Multi-document SummarizationUsing Wikinews Articles](https://www.researchgate.net/profile/Felipe_Bravo-Marquez/publication/233158220_A_Zipf-Like_Distant_Supervision_Approach_for_Multi-document_Summarization_Using_Wikinews_Articles/links/0fcfd509bfcdcf274f000000/A-Zipf-Like-Distant-Supervision-Approach-for-Multi-document-Summarization-Using-Wikinews-Articles.pdf)\
Perform extractive summarization on english Wikinews using Zipf-Like Distant Supervision (don't know how it works).
* [Multi-Topic Multi-Document Summarizer](https://arxiv.org/pdf/1401.0640.pdf)\
The paper mentions: "The data set is available in 7 languages including Arabic (http://www.nist.gov/tac/2011/Summarization/). It was derived from publicly available WikiNews English texts.". But I didn't find the dataset.
* [GameWikiSum: a Novel Large Multi-Document Summarization Dataset](https://arxiv.org/pdf/2002.06851.pdf) | [github](https://github.com/Diego999/GameWikiSum)\
Interesting work on video-game multi-document summarization which compare several methods (extractive and abstractive). Similar to what we want to do but for video-game documents.

### Multi-Lingual Summarization

* [Sequential Transfer Learning in NLP forGerman Text Summarization](http://ceur-ws.org/Vol-2458/paper8.pdf)\
Use BERT as encoder.
* [Abstract Text Summarization: A Low Resource Challenge](https://www.aclweb.org/anthology/D19-1616.pdf)\
Data augmentation for summarization. Use OpenNMT-py transformers.
* [MultiLing 2015: Multilingual Summarization of Single andMulti-Documents, On-line Fora, and Call-center Conversations](https://www.aclweb.org/anthology/W15-4638.pdf)\
They speach about a multilingual multi-Document Summarization dataset with 10 documents per topic and 15 topics. Again I didn't find it.
* [AllSummarizer system at MultiLing 2015:Multilingual single and multi-document summarization](https://www.aclweb.org/anthology/W15-4634.pdf) | [github](https://github.com/kariminf/AllSummarizer)\
Summarizer language and domain independent evaluate on  *MultiLing 2015 multi-document testing dataset* which seems great but I can't find it.

### English Abstract Summarization

* [BART: Denoising Sequence-to-Sequence Pre-training for NaturalLanguage Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) | [github](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)\
Great results on CNN-DM.
* [Exploring the Limits of Transfer Learning with aUnified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) | [github](https://github.com/google-research/text-to-text-transfer-transformer)\
Great results on CNN-DM.
* [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) | [github](https://github.com/nlpyang/PreSumm)\
Use BERT as encoder and a non-trained decoder. Two optimizers to avoid overfitting of BERT.
* [Transforming Wikipedia into Augmented Datafor Query-Focused Summarization](https://arxiv.org/pdf/1911.03324.pdf)\
/|Extractive summarization|\] Use Bert for topic based summary.

### Evaluation

* [EASY-M: Evaluation System for Multilingual Summarizers](https://www.aclweb.org/anthology/W19-89.pdf#page=63) | [web](https://summaryevaluation.azurewebsites.net/home)\
Online system that evaluate summaries with many metrics. Can be use to compare with our metrics.
