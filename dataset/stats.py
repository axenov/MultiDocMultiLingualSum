from utils import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_script_path",
    help="Path to the dataset script",
    type=str,
    default="./en_wiki_multi_news.py",
)
parser.add_argument(
    "--dataset_cache_path",
    help="Path to the cache folder",
    type=str,
    default="dataset/.en-wiki-multi-news-cache",
)
parser.add_argument(
    "--rouge",
    help="True if compute the ROUGE scores. Take a long time",
    type=bool,
    default=False,
)

args = parser.parse_args()

stats(args.dataset_script_path, args.dataset_cache_path, args.rouge)
