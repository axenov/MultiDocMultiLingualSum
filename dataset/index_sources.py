from utils import index_sources
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index_path", help='Path to the index file', type=str, default='dataset/sources/index/de.sources.index')
parser.add_argument("--json_path", help='Path to the json folder', type=str, default='dataset/wikinews/json.de')

args = parser.parse_args()

index_sources(args.json_path, args.index_path)