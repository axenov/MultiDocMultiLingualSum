import json
import pprint
from rouge_score import rouge_scorer
from random import choices
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from sources.utils import Library

from tqdm import tqdm

from nlp import load_dataset

def get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):

    if isinstance(ids, int):
        ids = [ids]
    
    docs = []
    for id in tqdm(ids, desc='Read wikinews articles'):

        with open(wikinews_json_path+'/{:06d}.json'.format(id), 'r') as json_file:
            try:
                docs.append(json.load(json_file))
            except:
                print(f'{id} doesn\'t exist')
                docs.append({})

    library = Library(sources_index_path, sources_json_path, sources_html_path)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    for i in tqdm(range(len(docs)), desc='Read sources'):
        sources_text = ""
        for j in range(len(docs[i]['sources'])):
            try:
                docs[i]['sources'][j] = library.get_json(docs[i]['sources'][j])
            except:
                print('source: {} doesn\'t exist'.format(docs[i]['sources'][j]))
                docs[i]['sources'][j] = {}
                continue
            if not 'maintext' in docs[i]['sources'][j].keys(): 
                continue
            if not isinstance(docs[i]['sources'][j]['maintext'], str): 
                docs[i]['sources'][j]['score'] = scorer.score('', '')
                continue
            sources_text += " " + docs[i]['sources'][j]['maintext']
            score = scorer.score(' '.join(docs[i]['text']), docs[i]['sources'][j]['maintext'])
            docs[i]['sources'][j]['score'] = score
        score = scorer.score(' '.join(docs[i]['text']), sources_text)
        docs[i]['score'] = score

    return docs

def print_docs(docs):
    for doc in docs:
        title = doc['title']
        text = ' '.join(doc['text'])
        sources = doc['sources']
        score = doc['score']
        print('=== {} ===\n'.format(title))
        print('ROUGE1 recall: {}'.format(score['rouge1'].recall))
        print('ROUGE2 recall: {}\n'.format(score['rouge2'].recall))
        print('--- summary ---\n')
        pprint.pprint(text)
        print('\n--- sources ---\n')
        for i, source in enumerate(sources):
            if source == {}: 
                print('*** Source not available ***\n')
                continue
            source_title = source['title']
            source_text = source['maintext']
            source_score = source['score']
            print('*** {} ***\n'.format(source_title))
            print('ROUGE1 recall: {}\n'.format(source_score['rouge1'].recall))
            pprint.pprint(source_text)
            print()
        print('======\n\n')

def store_docs(num, filename, wikinews_index_path, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):
    ids = get_ids(wikinews_index_path, num)
    docs = get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path)
    with open(filename, 'w') as f:
        for doc in docs:
            title = doc['title']
            text = ' '.join(doc['text'])
            sources = doc['sources']
            score = doc['score']
            f.write('=== {} ===\n\n'.format(title))
            f.write('ROUGE1 recall: {}\n'.format(score['rouge1'].recall))
            f.write('ROUGE2 recall: {}\n\n'.format(score['rouge2'].recall))
            f.write('--- summary ---\n\n')
            f.write(text)
            f.write('\n\n--- sources ---\n\n')
            for i, source in enumerate(sources):
                if source == {}: 
                    f.write('*** Source not available ***\n\n')
                    continue
                source_title = str(source['title'])
                source_text = str(source['maintext'])
                source_score = source['score']
                f.write('*** {} ***\n\n'.format(source_title))
                f.write('ROUGE1 recall: {}\n\n'.format(source_score['rouge1'].recall))
                f.write(source_text)
                f.write('\n\n')
            f.write('======\n\n')

def create_dataset(recall_threashold, hard, language, dataset_path, wikinews_index_path, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):
    ids = get_ids(wikinews_index_path)
    docs = get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path)
    num_sources = []
    dataset = []

    for doc in tqdm(docs, desc='Write dataset'):
        title = doc['title']
        summary = '\t'.join(doc['text'])
        sources = [source['maintext'] for source in doc['sources'] if isinstance(source, dict) and 'maintext' in source.keys() and source['maintext'] != None and source['language'] == language]
        if doc['score']['rouge1'].recall < recall_threashold:
            continue
        if len(sources) == 0:
            continue
        if not is_clean(summary, '|||'.join(sources), len(doc['sources']), hard=hard):
            continue
        num_sources.append(len(sources))
        entry = {'title': title, 'summary': summary, 'sources': '|||'.join(sources)}
        dataset.append(entry)

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.05, random_state=1)

    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.05/0.95, random_state=1)

    write_dataset(dataset_path, 'train', dataset_train)
    write_dataset(dataset_path, 'test', dataset_test)
    write_dataset(dataset_path, 'validation', dataset_val)

    print('number of article with:\n - 1 source: {}\n - 2 sources: {}\n - 3 sources: {}\n - 4 sources: {}\n - more sources: {}'.format(
        num_sources.count(1),
        num_sources.count(2),
        num_sources.count(3),
        num_sources.count(4),
        len(num_sources) - num_sources.count(1) - num_sources.count(2) - num_sources.count(3) - num_sources.count(4)
    ))

'''
take example of input and return True is the example is correct or False if it needs to be removed
'''
def is_clean(summary, sources, original_num_sources, hard):

    summary_NoW = len(word_tokenize(summary))
    document_NoW = len(word_tokenize(sources))
    num_sources = len(sources.split('|||'))

    # if summary is less than 1.5 times smaller than document
    if 1.5 * summary_NoW > document_NoW:
        return False

    if not hard:
        return True

    # if summary is less than 2 times smaller than document
    if 2 * summary_NoW > document_NoW:
        return False
    
    # if summary or document larger than 4000 words
    if summary_NoW > 4000 or document_NoW > 4000:
        return False

    # if more than half of sources was dead
    if num_sources < 0.5 * original_num_sources:
        return False

    return True

def write_dataset(dataset_path, type, dataset):
    with open(f'{dataset_path}/{type}.jsonl', 'w') as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write('\n')

def text_to_passages(text):
    def clean(string):
        string = string.replace('\n', '')
        string = string.replace('\t', '')
        return string
    passages = text.split('\n')
    passages = list(map(clean, passages))
    return [passage for passage in passages if passage != '']

def get_ids(wikinews_index_path, num=-1):
    ids = []
    with open(wikinews_index_path, 'r') as f:
        for line in f:
            elems = line.split('\t')
            if len(elems) != 2: continue
            ids.append(int(elems[1][:-1]))
    
    if num == -1:
        return ids
    
    return choices(ids, k=num)
    
def stats(dataset_script_path, dataset_cache_path):

    def list_stats(lst):
        lst = np.array(lst)
        d = scipy_stats.describe(lst)
        return {'mean': d.mean,
                'min': d.minmax[0],
                'max': d.minmax[1],
                'variance': d.variance}

    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
                    'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
                    'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}}
    num_sources = []
    sum_len = []

    def example_stats(example):

        # Rouge score
        rouge_score = scorer.score(example['summary'], example['document'])
        rouge_scores['rouge1']['precision'].append(rouge_score['rouge1'].precision)
        rouge_scores['rouge1']['recall'].append(rouge_score['rouge1'].recall)
        rouge_scores['rouge1']['fmeasure'].append(rouge_score['rouge1'].fmeasure)
        rouge_scores['rouge2']['precision'].append(rouge_score['rouge2'].precision)
        rouge_scores['rouge2']['recall'].append(rouge_score['rouge2'].recall)
        rouge_scores['rouge2']['fmeasure'].append(rouge_score['rouge2'].fmeasure)
        rouge_scores['rougeL']['precision'].append(rouge_score['rougeL'].precision)
        rouge_scores['rougeL']['recall'].append(rouge_score['rougeL'].recall)
        rouge_scores['rougeL']['fmeasure'].append(rouge_score['rougeL'].fmeasure)

        # Number of sources
        num = example['document'].count('|||') + 1
        num_sources.append(num)

        # Summary length
        sum_len.append(len(word_tokenize(example['summary'])))

    dataset = load_dataset(dataset_script_path, cache_dir=dataset_cache_path, split='train+test+validation')

    print(dataset)

    dataset.map(example_stats)

    print(len(num_sources), len(sum_len), len(rouge_scores['rouge1']['precision']))

    # Mean
    rouge_stats = {'rouge1': {'precision': list_stats(rouge_scores['rouge1']['precision']), 'recall': list_stats(rouge_scores['rouge1']['recall']), 'fmeasure': list_stats(rouge_scores['rouge1']['fmeasure'])},
                    'rouge2': {'precision': list_stats(rouge_scores['rouge2']['precision']), 'recall': list_stats(rouge_scores['rouge2']['recall']), 'fmeasure': list_stats(rouge_scores['rouge2']['fmeasure'])},
                    'rougeL': {'precision': list_stats(rouge_scores['rougeL']['precision']), 'recall': list_stats(rouge_scores['rougeL']['recall']), 'fmeasure': list_stats(rouge_scores['rougeL']['fmeasure'])}}
    num_sources_stats = {'num_sources': list_stats(num_sources), '1 source': num_sources.count(1), '2 sources': num_sources.count(2), '3 sources': num_sources.count(3), 'More sources': len(num_sources) - num_sources.count(1) - num_sources.count(2) - num_sources.count(3)}
    sum_len_stats = {'sum_len': list_stats(sum_len)}

    all_stats = {
        'rouge1 P': rouge_stats['rouge1']['precision'],
        'rouge1 R': rouge_stats['rouge1']['recall'],
        'rouge1 F': rouge_stats['rouge1']['fmeasure'],
        'rouge2 P': rouge_stats['rouge2']['precision'],
        'rouge2 R': rouge_stats['rouge2']['recall'],
        'rouge2 F': rouge_stats['rouge2']['fmeasure'],
        'rougeL P': rouge_stats['rougeL']['precision'],
        'rougeL R': rouge_stats['rougeL']['recall'],
        'rougeL F': rouge_stats['rougeL']['fmeasure'],
        'number of sources': list_stats(num_sources),
        'summary\'s number of words': list_stats(sum_len),
    }

    print('|       | mean | min | max | variance |')
    print('| --- | --- | --- | --- | --- |')
    for name, value in all_stats.items():
        print('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            name,
            value['mean'],
            value['min'],
            value['max'],
            value['variance']
        ))
    print()
    print('Source with:\n- 1 source: {}\n- 2 sources: {}\n- 3 sources: {}\n- More sources: {}\n'.format(
        num_sources_stats['1 source'],
        num_sources_stats['2 sources'],
        num_sources_stats['3 sources'],
        num_sources_stats['More sources']
    ))
    return rouge_stats, num_sources_stats, sum_len_stats

