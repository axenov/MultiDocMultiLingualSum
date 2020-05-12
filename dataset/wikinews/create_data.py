import argparse
from utils import get_pages_from_wiki_dump, read_index, write_index
import json
import os
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument("--wiki_dump_path", help='Path to the dump', type=str, default='dataset/wikinews/dumps/dewikinews-latest-pages-meta-current.xml.bz2')
parser.add_argument("--max_doc_count", help='Number of pages', type=int, default=0)
parser.add_argument("--index_path", help='Path to the index file (url id)', type=str, default='dataset/wikinews/index/de.wikinews.index')
parser.add_argument("--json_path", help='Path to the json folder', type=str, default='dataset/wikinews/json.de/')

args = parser.parse_args()

docs = get_pages_from_wiki_dump(args.wiki_dump_path, max_doc_count=args.max_doc_count)

index = read_index(args.index_path)
if index == {}:
    last_id = -1
else:
    last_id = max(list(index.values()))


json_path_template = '{:06d}.json'
index_template = '{}\t{:06d}\n'

new_index = {}

files = [join(args.json_path, f) for f in listdir(args.json_path) if isfile(join(args.json_path, f)) and join(args.json_path, f)[-4:] == 'json']
for filename in files:
    os.remove(filename)

for doc in docs:
    if doc['title'] in index.keys():
        id = index[doc['title']]
    else:
        last_id += 1
        id = last_id
    with open(args.json_path+json_path_template.format(id), 'w') as f:
        json.dump(doc, f, indent=4)
    new_index[doc['title']] = id

write_index(new_index, args.index_path)

