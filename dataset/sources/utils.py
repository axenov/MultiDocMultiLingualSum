from os import listdir
from os.path import isfile, join
import json

import pprint


def stats_sources(json_path):
    total = 0
    empty = 0
    with_maintext = 0
    without_maintext = 0
    languages = {}
    files = [
        join(json_path, f)
        for f in listdir(json_path)
        if isfile(join(json_path, f)) and join(json_path, f)[-4:] == "json"
    ]
    for filename in files:
        with open(filename) as json_file:
            doc = json.load(json_file)
        total += 1
        if doc == {}:
            empty += 1
            continue
        if "maintext" in doc.keys():
            if doc["maintext"] != None:
                with_maintext += 1
            else:
                without_maintext += 1
        if "language" in doc.keys():
            language = doc["language"]
            if language in languages.keys():
                languages[language] += 1
            else:
                languages[language] = 1

    return {
        "total": total,
        "empty": empty,
        "with_maintext": with_maintext,
        "without_maintext": without_maintext,
        "languages": languages,
    }


def print_stats(json_path):
    stats = stats_sources(json_path)
    pprint.pprint(stats, width=1)


class Library:
    def __init__(self, index_path, json_path, html_path):
        self.url2id = {}
        with open(index_path, "r") as f:
            for line in f:
                elems = line.split("\t")
                if len(elems) != 4:
                    continue
                self.url2id[elems[0]] = elems[3][:-1]

        self.json_template = json_path + "/{}.json"
        self.html_template = html_path + "/{}.html"

    def get_json(self, url):
        with open(self.json_template.format(self.url2id[url]), "r") as json_file:
            doc = json.load(json_file)
        return doc

    def get_html(self, url):
        with open(self.html_template.format(self.url2id[url]), "r") as f:
            html = f.read()
        return html
