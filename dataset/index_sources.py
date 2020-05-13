from utils import index_sources
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wikinews_json_path", help='Path to the wikinews json folder', type=str, default='dataset/wikinews/json.de')
parser.add_argument("--index_path", help='Path to the index file', type=str, default='dataset/sources/index/de.sources.index')
parser.add_argument("--html_path", help='Path to the html folder', type=str, default='dataset/sources/html')
parser.add_argument("--json_path", help='Path to the json folder', type=str, default='dataset/sources/json')

args = parser.parse_args()

index_sources(args.wikinews_json_path, args.index_path, args.html_path, args.json_path)