import argparse
from utils import get_pages_from_wiki_dump
import json

parser = argparse.ArgumentParser()
parser.add_argument("--wiki_dump_path", help='Path to the dump', type=str, default='dumps/dewikinews-latest-pages-meta-current.xml.bz2')
parser.add_argument("--max_doc_count", help='Number of pages', type=int, default=0)
parser.add_argument("--data_path", help='Path to the data .jsonl', type=str, default='data/data.jsonl')

args = parser.parse_args()

docs = get_pages_from_wiki_dump(args.wiki_dump_path, max_doc_count=args.max_doc_count)

with open(args.data_path, 'w') as fout:
    for doc in docs:
        fout.write(json.dumps(doc)) 
        fout.write('\n')