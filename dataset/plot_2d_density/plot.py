import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from transformers import BertTokenizer

from nlp import load_dataset

dataset = load_dataset('en_wiki_multi_news.py', cache_dir='dataset/.en-wiki-multi-news-cache', split='test+train+validation')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def tokenization_stats(summary, document, tokenizer):
    sum_toks = tokenizer.tokenize(summary.replace('\t', ' '))
    doc_toks = tokenizer.tokenize(document.replace('|||', ' '))
    return sum_toks, doc_toks

results = {'summary': [], 'document': []}

def stats(example):
    sum_toks, doc_toks = tokenization_stats(example['summary'], example['document'], tokenizer)
    results['summary'].append(len(sum_toks))
    results['document'].append(len(doc_toks))

dataset.map(stats)

data = []
for sum_l, doc_l in zip(results['summary'], results['document']):
        data.append([sum_l, doc_l])


x, y = np.array(data, dtype=float).T

sns_plot = sns.jointplot(x=x, y=y, kind="kde").set_axis_labels("# of toks in summary", "# of toks in document")
sns_plot.savefig("dataset/plot_2d_density/2d_density.png")

sns_plot = sns.jointplot(x=x, y=y, kind="kde", xlim=(0, 2000), ylim=(0, 10000)).set_axis_labels("# of toks in summary", "# of toks in document")
sns_plot.savefig("dataset/plot_2d_density/2d_density_zoom.png")

sns_plot = sns.jointplot(x=x, y=y, kind="kde", xlim=(0, 800), ylim=(0, 4000)).set_axis_labels("# of toks in summary", "# of toks in document")
sns_plot.savefig("dataset/plot_2d_density/2d_density_super_zoom.png")
