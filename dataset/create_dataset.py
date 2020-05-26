from utils import create_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help='Path to dataset file', type=str, default='dataset/en-wiki-multi-news')
parser.add_argument("--recall_threashold", help='recall_threashold', type=float, default=0.1)
parser.add_argument("--hard", help='True if hard cleaning', type=str, default=False)
parser.add_argument("--language", help='language of sources', type=str, default='en')
parser.add_argument("--wikinews_index_path", help='Path to the wikinews index file', type=str, default='dataset/wikinews/index/en.wikinews.index')
parser.add_argument("--wikinews_json_path", help='Path to the wikinews json folder', type=str, default='dataset/wikinews/json.en')
parser.add_argument("--sources_index_path", help='Path to the index file', type=str, default='dataset/sources/index/en.sources.index')
parser.add_argument("--sources_html_path", help='Path to the html folder', type=str, default='dataset/sources/html')
parser.add_argument("--sources_json_path", help='Path to the json folder', type=str, default='dataset/sources/json')

args = parser.parse_args()

create_dataset(args.recall_threashold, args.hard, args.language, args.dataset_path, args.wikinews_index_path, args.wikinews_json_path, args.sources_index_path, args.sources_json_path, args.sources_html_path)
