from utils_index_sources import index_sources
import argparse
import os
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wikinews_json_path",
    help="Path to the wikinews json folder",
    type=str,
    default="dataset/wikinews/json.fr",
)
parser.add_argument(
    "--index_path",
    help="Path to the index file",
    type=str,
    default="dataset/sources/index/fr.sources.index",
)
parser.add_argument(
    "--html_path",
    help="Path to the html folder",
    type=str,
    default="dataset/sources/html",
)
parser.add_argument(
    "--json_path",
    help="Path to the json folder",
    type=str,
    default="dataset/sources/json",
)
parser.add_argument(
    "--max_url_count",
    help="Number of sources to index. Set to -1 for all sources",
    type=int,
    default=-1,
)
parser.add_argument("--max_workers", help="Number of max_workers", type=int, default=6)
parser.add_argument(
    "--restart",
    help="True if you want to restart the indexing",
    type=bool,
    default=False,
)

args = parser.parse_args()

if not os.path.exists(args.index_path):
    os.system("touch {}".format(args.index_path))

assert os.path.exists(args.html_path), "html path does not exist"
assert os.path.exists(args.json_path), "json path does not exist"

if args.restart:
    print("Remove all previus indexing")
    os.system('for i in {}/*.json; do rm "$i"; done'.format(args.json_path))
    os.system('for i in {}/*.html; do rm "$i"; done'.format(args.html_path))
    os.remove(args.index_path)
    os.system("touch {}".format(args.index_path))


loop = asyncio.get_event_loop()
future = asyncio.ensure_future(
    index_sources(
        args.wikinews_json_path,
        args.index_path,
        args.html_path,
        args.json_path,
        args.max_url_count,
        args.max_workers,
        args.restart,
    )
)
loop.run_until_complete(future)
