# Dataset preparation

## Clone repository 

```
git clone git clone https://github.com/airKlizz/MultiDocMultiLingualSum
cd MultiDocMultiLingualSum/
```

## Download Wikinews dump

To download the german Wikinews dump run:

```
cd dataset/wikinews/dumps/
wget https://dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-meta-current.xml.bz2
cd ../../..
```

## Create json files

Json files contain one article stored in json format as follows:

```json
{
  "title": "title of the article", 
  "text": "list of paragraphs", 
  "categories": ["categorie 1", "categorie 2", "..."] ,
  "sources": ["sources 1", "sources 2", "..."] 
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

## Index sources

Extract html and content from source urls of Wikinews articles. The script uses the [archive](https://web.archive.org/) version of the page if it exists otherwise it archives the page.

```
python dataset/index_sources.py --wikinews_json_path 'dataset/wikinews/json.en' \
                                --index_path 'dataset/sources/index/en.sources.index' \
                                --html_path 'dataset/sources/html' \
                                --json_path 'dataset/sources/json' \
                                --max_url_count -1 \
                                --max_workers 10
```

> Remarks: \
> The script raises an error and stops. We just have to rerun it using the same command.\
> If you want to restart the sources indexing, just add ``--restart True`` to the command. It will remove all previus files created.\
> The script takes more than 24 hours, depending to your internet connection. You can split the time by executing multiple times the script with the argument ``--max_url_count`` set to the number of sources you want to index.\
> The script will create 3 logs files: ``work_done.log``, ``ok.log`` and ``error.log`` to the racine of the project. They contains respectively the number of urls processed and the number remaining, if url was ok or not for each sources and the errors.

## Create dataset

Link Wikinews articles with sources that have been indexed to create a dataset. The dataset is composed of 3 ``.jsonl`` files. ``train.jsonl``, ``validation.jsonl`` and ``test.jsonl`` splitted with a ratio 0.9:0.05:0.05. Each line of these ``.jsonl`` files are a json object: 

```json
{
  "title": "title of the article", 
  "summary": "paragraphs1\tparagraphs2\t...",
  "sources": "sources 1|||sources 2|||..."
}
```

To create the dataset, you have to run:

```
python dataset/create_dataset.py --dataset_path "dataset/en-wiki-multi-news"\
                                --recall_threashold 0.1\
                                --language "en"\
                                --wikinews_index_path "dataset/wikinews/index/en.wikinews.index"\
                                --wikinews_json_path "dataset/wikinews/json.en"\
                                --sources_index_path "dataset/sources/index/en.sources.index"\
                                --sources_html_path "dataset/sources/html"\
                                --sources_json_path "dataset/sources/json"
```

## Use the dataset 

The easiest way to use the dataset is to load it using the [nlp](https://github.com/huggingface/nlp) library from [huggingface](https://huggingface.co/).

```python
from nlp import load_dataset

dataset = load_dataset('dataset/en-wiki-multi-news/en_wiki_multi_news.py', cache_dir='dataset/.en-wiki-multi-news-cache')

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']
```

## Stats

|       | mean | min | max | variance |
| --- | --- | --- | --- | --- |
| rouge1 P | 0.341 | 0.047 | 0.644 | 0.035 |
| rouge1 R | 0.761 | 0.385 | 0.967 | 0.027 |
| rouge1 F | 0.432 | 0.089 | 0.718 | 0.037 |
| rouge2 P | 0.183 | 0.031 | 0.493 | 0.023 |
| rouge2 R | 0.401 | 0.088 | 0.829 | 0.036 |
| rouge2 F | 0.231 | 0.057 | 0.618 | 0.029 |
| rougeL P | 0.204 | 0.034 | 0.498 | 0.025 |
| rougeL R | 0.439 | 0.175 | 0.838 | 0.031 |
| rougeL F | 0.255 | 0.063 | 0.625 | 0.029 |
| number of sources | 1.000 | 1.000 | 1.000 | 0.000 |
| summary's number of words | 1568.318 | 171.000 | 8554.000 | 4242442.037 |