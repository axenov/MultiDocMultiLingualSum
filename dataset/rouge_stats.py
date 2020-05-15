from utils import rouge_stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wikinews_index_path", help='Path to the wikinews index file', type=str, default='dataset/wikinews/index/de.wikinews.index')
parser.add_argument("--wikinews_json_path", help='Path to the wikinews json folder', type=str, default='dataset/wikinews/json.de')
parser.add_argument("--sources_index_path", help='Path to the index file', type=str, default='dataset/sources/index/de.sources.index')
parser.add_argument("--sources_html_path", help='Path to the html folder', type=str, default='dataset/sources/html.de')
parser.add_argument("--sources_json_path", help='Path to the json folder', type=str, default='dataset/sources/json.de')

args = parser.parse_args()

rouge_stats(args.wikinews_index_path, args.wikinews_json_path, args.sources_index_path, args.sources_json_path, args.sources_html_path)
