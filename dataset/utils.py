import json
import pprint
from rouge_score import rouge_scorer
from random import choices
import pandas as pd

from sources.utils import Library

from tqdm import tqdm

def get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):

    if isinstance(ids, int):
        ids = [ids]
    
    docs = []
    for id in tqdm(ids, desc='Read wikinews articles'):

        with open(wikinews_json_path+'/{:06d}.json'.format(id), 'r') as json_file:
            docs.append(json.load(json_file))

    library = Library(sources_index_path, sources_json_path, sources_html_path)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    for i in tqdm(range(len(docs)), desc='Read sources'):
        sources_text = ""
        for j in range(len(docs[i]['sources'])):
            docs[i]['sources'][j] = library.get_json(docs[i]['sources'][j])
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

def create_dataset(recall_threashold, language, dataset_path, wikinews_index_path, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):
    ids = get_ids(wikinews_index_path)
    docs = get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path)
    num_sources = []
    with open(dataset_path, 'w') as f:
        for doc in tqdm(docs, desc='Write dataset'):
            summary = doc['text']
            sources = [source['maintext'] for source in doc['sources'] if 'maintext' in source.keys() and source['maintext'] != None and source['language'] == language]
            if len(sources) == 0:
                continue
            num_sources.append(len(sources))
            sources = list(map(text_to_passages, sources))
            entry = {'summary': summary, 'sources': sources}
            json.dump(entry, f)
            f.write('\n')

    print('number of article with:\n - 1 source: {}\n - 2 sources: {}\n - 3 sources: {}\n - 4 sources: {}\n - more sources: {}'.format(
        num_sources.count(1),
        num_sources.count(2),
        num_sources.count(3),
        num_sources.count(4),
        len(num_sources) - num_sources.count(1) - num_sources.count(2) - num_sources.count(3) - num_sources.count(4)
    ))

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
    
def rouge_stats(wikinews_index_path, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path):
    ids = get_ids(wikinews_index_path)
    docs = get_wikinews(ids, wikinews_json_path, sources_index_path, sources_json_path, sources_html_path)

    rouge1_recall = []
    rouge2_recall = []

    for doc in docs:
        rouge1_recall.append(doc['score']['rouge1'].recall)
        rouge2_recall.append(doc['score']['rouge2'].recall)

    data = {'rouge1_recall': rouge1_recall, 'rouge2_recall': rouge2_recall}
    df = pd.DataFrame(data=data)
    print(df.describe())