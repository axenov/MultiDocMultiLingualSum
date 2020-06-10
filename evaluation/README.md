# Summarization Baselines



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

This repository is based on [``nlp`` library](https://github.com/huggingface/nlp) for load data and to compute ROUGE metric.

The idea is that you have a summarization dataset (``nlp.Dataset`` class) with at least a column with texts to summarize (``document_column_name``) and one column with reference summaries (``summary_colunm_name``). Then you want to run multiple baselines on it and compare ROUGE results of these differents methods of summarization.

You can add your summarization model (extractive or abstractive) as a new baseline to compare its performance with other baselines. Go [here](#add-baseline) for more details to add a baseline.

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

### Run ``python run_baseline --run_args_file "args/run_args_extractives.jon"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 38.84% | 45.03% | 39.13% | 12.81% | 14.77% | 12.85% | 17.13% | 20.45% | 17.42% |
| Lead | 40.56% | 49.35% | 41.85% | 15.62% | 18.78% | 16.01% | 20.91% | 25.75% | 21.65% |
| LexRank | 34.05% | 53.30% | 39.40% | 12.63% | 19.83% | 14.62% | 16.04% | 26.00% | 18.79% |
| TextRank | 34.60% | 51.97% | 39.31% | 12.55% | 19.04% | 14.33% | 16.24% | 25.20% | 18.65% |
| TF-IDF | 36.72% | 50.81% | 40.28% | 12.75% | 18.06% | 14.09% | 17.01% | 24.46% | 18.91% |
| Rouge Oracle | 50.48% | 56.48% | 49.54% | 28.52% | 29.93% | 27.07% | 22.68% | 25.83% | 22.29% |

### Run ``python run_baseline --run_args_file "args/run_args_abstractives_with_lead.jon"``

|     | rouge1 P |  rouge1 R | rouge1 F | rouge2 P |  rouge2 R | rouge2 F | rougeL P |  rougeL R | rougeL F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lead + Bart | 44.93% | 47.51% | 43.78% | 17.30% | 18.07% | 16.75% | 22.72% | 24.52% | 22.28% |
| Lead + Bart-cnn | 46.48% | 46.19% | 43.92% | 18.54% | 18.05% | 17.35% | 24.48% | 24.47% | 23.16% |
| Lead + T5 | 48.01% | 40.61% | 41.60% | 18.96% | 15.75% | 16.30% | 26.14% | 22.20% | 22.63% |
| Lead + T5 with title | 47.90% | 40.09% | 41.30% | 18.65% | 15.18% | 15.86% | 25.80% | 21.69% | 22.27% |
