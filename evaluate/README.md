# Summarization Baselines

Implementation a varius summarization baselines including. This repository can be used to compare ROUGE results of summarization method (extractive or abstractive). 

This repository is based on [``nlp`` library](https://github.com/huggingface/nlp) for load data and to compute ROUGE metric.

The idea is that you have a summarization dataset (``nlp.Dataset`` class) with at least a column with texts to summarize (``document_column_name``) and one column with reference summaries (``summary_colunm_name``). Then you want to run multiple baselines on it and compare ROUGE results of these differents methods of summarization. 

See available baseline [here](#available-baselines). You can add your summarization model (extractive or abstractive) as a new baseline to compare its performance with other baselines. Go [here](#add-baseline) for more details to add a baseline.

## Available baselines

- Random: Select n sentences randomly.
- Lead: Select the n first sentences.
- LexRank: Compute similarity between sentences using TF-IDF and select the n first sentences ranked using PageRank style algorithm.
- TextRank: Compute similarity between sentences using containing words and select the n first sentences ranked using PageRank style algorithm.
- TF-IDF: Compute similarity between sentences and the title of the article using TF-IDF and select the n first sentences based on the similarity score.
- Oracle: **Cheating method**. Maximize a ROUGE score. 
- Bert2Bert: Transformer model. Implementation thanks to [huggingface](https://huggingface.co/).
- Bart: Transformer model. Implementation thanks to [huggingface](https://huggingface.co/).
- T5: Transformer model. Implementation thanks to [huggingface](https://huggingface.co/).
- Combine: Allow to combine an extractive baseline with an extractive one. Example TextRank with Bart.

## Usage

To run baseline you first have to configure a args ``.json`` file with your parameters. See [here](#args-file) to see how the ``.json`` file is built.

Once you have all baselines you need, your dataset and your configured ``run_args.json`` file, you can run the computation by running:

```bash
python run_baseline.py --run_args_file "path/to/run_args.json"
```

Results are stored to the files/folder you put in the ``run_args.json`` file.

## Add baseline

If you want to add your baseline you have to create a script similar to ``baselines/lead.py`` for extractive baseline or ``baselines/bart.py`` for abstractive baseline which contain a subclass of ``Baseline`` and define the function ``def rank_sentences(self, dataset, document_column_name, **kwargs)`` or ``def get_summaries(self, dataset, document_column_name, **kwargs)``. 

For extractive baseline, the function ``rank_sentences`` ranks all sentences of each document and add scores and sentences in a new column of the dataset. It returns the dataset.

For abstractive baseline, the function ``get_summaries`` summaries each document and add summaries (also called hypotheses) in a new column of the dataset. It returns the dataset.

Then just add you baseline on the ``baselines/baselines.py`` file by adding a ``if`` and you can use your baseline.

## Args file

This is an example of a ``run_args.json`` file:

```json
{
    "baselines": [
        {"baseline_class": "Random", "init_kwargs": {"name": "Random"}, "run_kwargs": {"num_sentences": 10}},
        {"baseline_class": "Lead", "init_kwargs": {"name": "Lead"}, "run_kwargs": {"num_sentences": 10}},
        {"baseline_class": "LexRank", "init_kwargs": {"name": "LexRank"}, "run_kwargs": {"num_sentences": 10, "threshold": 0.03, "increase_power": true}},
        {
            "baseline_class": "Bart", 
            "init_kwargs": {
                "name": "Bart CNN",
                "model_name": "bart-large-cnn",
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
            }
        },
        {
            "baseline_class": "T5", 
            "init_kwargs": {
                "name": "T5 base",
                "model_name": "t5-base",
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
            }
        },
        {
            "baseline_class": "T5", 
            "init_kwargs": {
                "name": "T5 fine tuned",
                "model_name": ["t5-base", "/content/drive/My Drive/Colab Notebooks/Multi-wiki-news/English/t5-wild-glitter-2"],
                "input_max_length": 512,
                "device": "cuda",
                "batch_size": 8
            }, 
            "run_kwargs": {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 400,
                "min_length": 200,
                "no_repeat_ngram_size": 3,
                "early_stopping": true
            }
        }
    ],

    "dataset": {
        "name": "en_wiki_multi_news_cleaned.py",
        "split": "test",
        "cache_dir": ".en-wiki-multi-news-cache",
        "document_column_name": "document",
        "summary_colunm_name": "summary"
    },
    "run": {
        "hypotheses_folder": "hypotheses/",
        "csv_file": "results.csv",
        "md_file": "results.md",
        "rouge_types": {
            "rouge1": ["mid.fmeasure"],
            "rouge2": ["mid.fmeasure"],
            "rougeL": ["mid.fmeasure"]
        }
    }
}
```

The file is composed of 3 arguments:

- ``baselines``: it defines all baselines you want to compare with for each the associate ``class``, ``init_kwargs`` which are arguments pass to the ``init`` function of the ``class`` and ``run_kwargs`` which are arguments pass to the run function,
- ``dataset``: it defines dataset's arguments with the ``name`` which is the name of the ``nlp`` dataset or the path to the dataset python script, the ``split`` and the ``cache_dir`` of the dataset (see [nlp](https://github.com/huggingface/nlp) ``load_dataset`` function), ``document_column_name`` which is the name of the column in the dataset containing the texts to summarize and ``summary_column_name`` which is the name of the column in the dataset containing the summaries,
- ``run``: it defines the ROUGE run arguments with the ``folder`` to save hypotheses, optionnal ``csv_file`` and ``md_file`` to save results to the corresponding format and ``rouge_types`` which are the type of ROUGE scores to compute (see [nlp](https://github.com/huggingface/nlp) ``rouge`` metric). 

See ``args/`` to see more examples.

## Results for ``en-wiki-multi-news``

### Extractives methods

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_en_extractives.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 38.90% | 45.02% | 39.16% | 12.85% | 14.81% | 12.89% | 17.14% | 20.43% | 17.41% |
| Lead | 40.61% | 49.29% | 41.85% | 15.66% | 18.78% | 16.01% | 20.88% | 25.72% | 21.61% |
| LexRank | 36.59% | 53.35% | 41.20% | 13.74% | 19.95% | 15.41% | 16.55% | 25.09% | 18.89% |
| TextRank | 36.29% | 51.95% | 40.55% | 13.36% | 19.04% | 14.91% | 16.58% | 24.52% | 18.74% |
| LexRankSumy | 38.40% | 51.67% | 41.80% | 13.66% | 18.27% | 14.78% | 16.83% | 23.49% | 18.55% |
| TextRankSumy | 33.80% | 54.67% | 39.67% | 12.61% | 20.12% | 14.72% | 15.29% | 25.62% | 18.16% |
| TF-IDF | 37.02% | 52.19% | 41.05% | 13.14% | 18.56% | 14.55% | 16.89% | 24.65% | 18.96% |
| Rouge Oracle | 50.49% | 56.47% | 49.57% | 28.59% | 29.94% | 27.05% | 22.64% | 25.79% | 22.28% |

### Abstractives methods combined with Lead

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_en_abstractives_with_lead.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 44.84% | 47.52% | 43.74% | 17.31% | 18.06% | 16.73% | 22.76% | 24.56% | 22.33% |
| Lead + Bart-cnn | 46.49% | 46.20% | 43.93% | 18.55% | 18.06% | 17.35% | 24.46% | 24.49% | 23.12% |
| Lead + T5 | 48.02% | 40.68% | 41.66% | 18.93% | 15.77% | 16.28% | 26.14% | 22.19% | 22.65% |
| Lead + T5 with title | 49.07% | 40.56% | 41.95% | 19.32% | 15.71% | 16.35% | 26.77% | 22.31% | 22.92% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_lead_on_en.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 44.46% | 47.42% | 43.59% | 17.04% | 17.86% | 16.54% | 22.55% | 24.48% | 22.24% |
| Lead + T5 | 48.41% | 39.33% | 41.04% | 18.72% | 14.82% | 15.67% | 26.02% | 21.30% | 22.11% |

### Abstractives methods combined with Oracle

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_en_abstractives_with_oracle.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 47.32% | 49.52% | 45.98% | 20.61% | 20.75% | 19.61% | 24.71% | 26.24% | 24.11% |
| RougeOracle + Bart-cnn | 48.67% | 47.83% | 45.75% | 21.42% | 20.27% | 19.69% | 25.84% | 25.62% | 24.30% |
| RougeOracle + T5 | 54.84% | 44.81% | 46.66% | 27.20% | 20.99% | 22.42% | 30.34% | 24.67% | 25.69% |
| RougeOracle + T5 with title | 56.13% | 44.37% | 46.94% | 27.99% | 20.79% | 22.58% | 31.78% | 24.78% | 26.25% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_oracle_on_en.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 47.17% | 49.75% | 45.98% | 20.50% | 20.93% | 19.61% | 24.72% | 26.32% | 24.09% |
| RougeOracle + T5 | 55.83% | 43.69% | 46.43% | 27.80% | 20.26% | 22.19% | 31.20% | 24.21% | 25.75% |

## Results for ``de-wiki-multi-news``

### Extractives methods

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_de_extractives.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 32.87% | 35.62% | 31.49% | 8.40% | 9.27% | 8.11% | 13.87% | 15.31% | 13.35% |
| Lead | 36.14% | 38.68% | 34.53% | 10.71% | 11.85% | 10.40% | 16.67% | 18.55% | 16.17% |
| LexRank | 32.25% | 43.67% | 34.81% | 9.78% | 13.43% | 10.64% | 13.77% | 19.41% | 15.11% |
| TextRank | 32.73% | 40.48% | 33.44% | 9.47% | 11.99% | 9.84% | 14.13% | 17.84% | 14.50% |
| LexRankSumy | 34.67% | 41.37% | 35.11% | 10.14% | 12.39% | 10.36% | 14.44% | 18.02% | 14.88% |
| TextRankSumy | 30.51% | 45.56% | 34.15% | 9.28% | 14.10% | 10.49% | 12.70% | 19.83% | 14.46% |
| TF-IDF | 33.22% | 39.89% | 34.04% | 9.68% | 11.92% | 10.03% | 14.25% | 17.80% | 14.82% |
| Rouge Oracle | 44.62% | 46.51% | 41.84% | 20.87% | 20.56% | 19.03% | 19.02% | 20.47% | 18.05% |

### Abstractives methods combined with Lead

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_abstractives_with_lead.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 45.42% | 26.90% | 31.39% | 13.59% | 8.09% | 9.39% | 23.00% | 13.93% | 16.02% |
| Lead + T5 | 41.91% | 28.34% | 31.31% | 12.10% | 8.26% | 9.09% | 21.01% | 14.38% | 15.76% |
| Lead + T5 with title | 42.19% | 28.70% | 31.60% | 12.52% | 8.63% | 9.43% | 21.25% | 14.70% | 16.03% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_lead_on_de.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 43.63% | 28.13% | 31.79% | 13.01% | 8.45% | 9.49% | 21.85% | 14.38% | 16.09% |
| Lead + T5 | 41.57% | 28.90% | 31.52% | 12.30% | 8.59% | 9.37% | 20.83% | 14.72% | 15.91% |

### Abstractives methods combined with Oracle

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_abstractives_with_oracle.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 46.58% | 27.31% | 31.95% | 14.74% | 8.49% | 9.93% | 23.63% | 14.13% | 16.33% |
| RougeOracle + T5 | 47.06% | 31.02% | 34.57% | 17.14% | 10.71% | 12.20% | 24.52% | 16.18% | 18.02% |
| RougeOracle + T5 with title | 47.19% | 31.36% | 34.88% | 17.27% | 10.92% | 12.39% | 24.42% | 16.36% | 18.09% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_oracle_on_de.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 44.99% | 28.71% | 32.59% | 14.57% | 9.05% | 10.35% | 22.90% | 14.86% | 16.71% |
| RougeOracle + T5 | 46.52% | 31.59% | 34.89% | 17.09% | 10.94% | 12.36% | 23.88% | 16.37% | 17.95% |

## Results for ``fr-wiki-multi-news``

### Extractives methods

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_extractives.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 22.82% | 49.41% | 28.34% | 8.11% | 18.44% | 10.27% | 11.06% | 26.33% | 14.11% |
| Lead | 23.15% | 52.00% | 29.12% | 9.14% | 21.42% | 11.72% | 12.23% | 30.23% | 15.91% |
| LexRank | 25.26% | 46.36% | 29.78% | 9.53% | 17.12% | 11.19% | 12.62% | 24.74% | 15.22% |
| TextRank | 25.32% | 46.51% | 29.81% | 9.32% | 17.13% | 11.01% | 12.69% | 25.05% | 15.28% |
| LexRankSumy | 26.79% | 44.36% | 30.07% | 9.32% | 14.97% | 10.34% | 12.89% | 22.77% | 14.78% |
| TextRankSumy | 23.69% | 48.26% | 29.10% | 8.85% | 17.61% | 10.87% | 11.71% | 25.45% | 14.69% |
| TF-IDF | 25.95% | 45.72% | 29.91% | 9.16% | 16.12% | 10.61% | 12.85% | 24.15% | 15.09% |
| Rouge Oracle | 31.88% | 50.79% | 34.25% | 15.91% | 24.33% | 16.64% | 15.90% | 27.70% | 17.57% |

### Abstractives methods combined with Lead

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_abstractives_with_lead.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 44.79% | 41.61% | 37.83% | 18.69% | 18.76% | 16.06% | 25.71% | 25.85% | 22.25% |
| Lead + T5 | 42.63% | 40.97% | 37.06% | 17.68% | 18.49% | 15.98% | 24.84% | 25.57% | 22.28% |
| Lead + T5 with title | 41.80% | 43.74% | 37.27% | 17.45% | 20.27% | 16.00% | 24.17% | 27.70% | 22.21% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_lead_on_fr.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 38.61% | 43.30% | 36.50% | 15.82% | 19.55% | 15.45% | 21.37% | 26.61% | 21.01% |
| Lead + T5 with prefix in fr | 38.38% | 39.09% | 34.07% | 16.08% | 17.91% | 14.54% | 22.36% | 24.62% | 20.28% |
| Lead + T5 with prefix in en | 40.85% | 37.90% | 35.53% | 18.18% | 17.59% | 16.34% | 24.57% | 24.05% | 22.04% |

### Abstractives methods combined with Oracle

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_abstractives_with_oracle.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 46.15% | 41.76% | 38.30% | 20.01% | 19.13% | 16.65% | 26.26% | 25.91% | 22.41% |
| RougeOracle + T5 | 44.84% | 42.21% | 38.51% | 19.89% | 19.96% | 17.56% | 26.21% | 26.73% | 23.34% |
| RougeOracle + T5 with title | 43.12% | 45.00% | 38.50% | 19.05% | 21.49% | 17.26% | 24.98% | 28.37% | 22.90% |

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_combine_abstractives_with_oracle_on_fr.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RougeOracle + Bart | 39.58% | 43.73% | 36.99% | 16.74% | 20.02% | 16.01% | 22.10% | 26.73% | 21.36% |
| RougeOracle + T5 | 40.31% | 40.95% | 35.75% | 18.16% | 19.71% | 16.28% | 23.54% | 25.89% | 21.38% |

## Results for ``multi-news``

### Abstractives methods combined with Lead

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_en_abstractives_with_lead_on_multi_news.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart-cnn | 42.32% | 47.39% | 43.61% | 15.08% | 17.08% | 15.63% | 20.91% | 23.77% | 21.68% |
