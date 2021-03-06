import argparse
from utils import get_pages_from_wiki_dump, write_index
import json
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wiki_dump_path",
    help="Path to the dump",
    type=str,
    default="dataset/wikinews/dumps/frwikinews-latest-pages-meta-current.xml.bz2",
)
parser.add_argument("--max_doc_count", help="Number of pages", type=int, default=0)
parser.add_argument(
    "--index_path",
    help="Path to the index file (url id)",
    type=str,
    default="dataset/wikinews/index/fr.wikinews.index",
)
parser.add_argument(
    "--json_path",
    help="Path to the json folder",
    type=str,
    default="dataset/wikinews/json.fr/",
)

args = parser.parse_args()

docs = get_pages_from_wiki_dump(args.wiki_dump_path, max_doc_count=args.max_doc_count)

id = -1
json_path_template = "{:06d}.json"
index_template = "{}\t{:06d}\n"

Path(args.json_path).mkdir(parents=False, exist_ok=True)
new_index = {}
files = [
    join(args.json_path, f)
    for f in listdir(args.json_path)
    if isfile(join(args.json_path, f)) and join(args.json_path, f)[-4:] == "json"
]
for filename in files:
    os.remove(filename)

for doc in docs:
    id += 1
    with open(args.json_path + json_path_template.format(id), "w") as f:
        json.dump(doc, f, indent=4)
    new_index[doc["title"]] = id

write_index(new_index, args.index_path)
