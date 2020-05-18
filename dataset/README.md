# Dataset preparation

The following commands are to create the english dataset. To create the german one, run the same commands with removing all 'en' by 'de'.

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
| rouge1 P | 0.251 | 0.002 | 1.000 | 0.027 |
| rouge1 R | 0.737 | 0.014 | 1.000 | 0.034 |
| rouge1 F | 0.328 | 0.003 | 0.995 | 0.023 |
| rouge2 P | 0.115 | 0.000 | 0.990 | 0.010 |
| rouge2 R | 0.357 | 0.000 | 1.000 | 0.034 |
| rouge2 F | 0.154 | 0.000 | 0.987 | 0.012 |
| rougeL P | 0.134 | 0.001 | 0.993 | 0.011 |
| rougeL R | 0.400 | 0.014 | 1.000 | 0.024 |
| rougeL F | 0.175 | 0.002 | 0.990 | 0.010 |
| number of sources | 2.553 | 1.000 | 43.000 | 4.166 |
| summary's number of words | 356.144 | 3.000 | 13281.000 | 117325.758 |

Source with:
- 1 source: 4327
- 2 sources: 4361
- 3 sources: 2607
- More sources: 2629

Run ``python dataset/stats.py`` to reproduce.
