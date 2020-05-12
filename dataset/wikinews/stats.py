from utils import stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", help='Path to the csv', type=str, default='stats.csv')
parser.add_argument("--json_path", help='Path to the data .jsonl', type=str, default='dataset/wikinews/json.de')
parser.add_argument("--save_csv", help='True if you want to save the stats csv', type=bool, default=False)

args = parser.parse_args()

stats(args.json_path, args.csv_path, args.save_csv)