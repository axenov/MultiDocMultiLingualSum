import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help='Path to the data .jsonl', type=str, default='data/data.jsonl')

args = parser.parse_args()

docs = []
with open(args.data_path, 'r') as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

num_docs = len(docs)

while True:
    try:
        idx = int(input('Enter the doc you want to see (random: -1, max: {}, stop: -2) : '.format(num_docs-1)))
    except:
        print('Enter an integer.')
        continue
    if idx == -2: 
        break
    if idx >= num_docs: 
        print('Enter an coorect integer.')
        continue
    if idx == -1:
        idx = random.randint(0, num_docs-1)
    print('\n\nTitle: {}\n\n{}\n\nSource: {}\n\n'.format(docs[idx]['title'], '\n'.join(docs[idx]['text']), ' - '.join(docs[idx]['sources'])))
    