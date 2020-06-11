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
| Random | 38.84% | 45.03% | 39.13% | 12.81% | 14.77% | 12.85% | 17.13% | 20.45% | 17.42% |
| Lead | 40.56% | 49.35% | 41.85% | 15.62% | 18.78% | 16.01% | 20.91% | 25.75% | 21.65% |
| LexRank | 34.05% | 53.30% | 39.40% | 12.63% | 19.83% | 14.62% | 16.04% | 26.00% | 18.79% |
| TextRank | 34.60% | 51.97% | 39.31% | 12.55% | 19.04% | 14.33% | 16.24% | 25.20% | 18.65% |
| TF-IDF | 36.72% | 50.81% | 40.28% | 12.75% | 18.06% | 14.09% | 17.01% | 24.46% | 18.91% |
| Rouge Oracle | 50.48% | 56.48% | 49.54% | 28.52% | 29.93% | 27.07% | 22.68% | 25.83% | 22.29% |

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

## Results for ``de-wiki-multi-news``

### Extractives methods

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_de_extractives.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 32.89% | 35.59% | 31.52% | 8.38% | 9.29% | 8.11% | 13.87% | 15.28% | 13.35% |
| Lead | 36.08% | 38.67% | 34.45% | 10.71% | 11.82% | 10.38% | 16.67% | 18.59% | 16.17% |
| LexRank | 30.46% | 43.29% | 33.53% | 9.22% | 13.53% | 10.30% | 13.25% | 19.75% | 14.85% |
| TextRank | 31.30% | 40.23% | 32.52% | 8.93% | 11.81% | 9.42% | 13.63% | 18.10% | 14.33% |
| TF-IDF | 33.13% | 40.35% | 33.77% | 9.75% | 12.29% | 10.05% | 14.58% | 18.57% | 15.12% |
| Rouge Oracle | 44.57% | 46.50% | 41.78% | 20.88% | 20.57% | 19.04% | 19.02% | 20.48% | 18.08% |

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

## Results for ``fr-wiki-multi-news``

### Extractives methods

Run ``python evaluate/run_baseline.py --run_args_file "evaluate/args/run_args_fr_extractives.json"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 26.49% | 41.97% | 28.60% | 8.73% | 14.60% | 9.62% | 13.06% | 22.54% | 14.50% |
| Lead | 27.40% | 46.05% | 30.59% | 10.42% | 18.61% | 11.90% | 14.52% | 26.89% | 16.82% |
| LexRank | 23.11% | 49.16% | 28.70% | 8.93% | 19.39% | 11.22% | 11.66% | 27.28% | 14.94% |
| TextRank | 23.36% | 48.41% | 28.70% | 8.70% | 18.73% | 10.85% | 11.77% | 26.57% | 14.86% |
| TF-IDF | 26.03% | 48.39% | 30.22% | 9.65% | 19.21% | 11.52% | 13.26% | 27.29% | 15.99% |
| Rouge Oracle | 25.75% | 56.34% | 31.69% | 12.42% | 26.55% | 15.06% | 12.31% | 30.16% | 15.71% |

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
