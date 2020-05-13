from os import listdir
from os.path import isfile, join

import json
import requests
import re
import savepagenow
from tqdm import tqdm

from newsplease import NewsPlease
import newspaper

def request(url):
    html, archive_url, ok = get_archive(url)
    if ok:
        article = extract(html)
    else:
        article = {}       
    return html, archive_url, article, ok

def get_archive(url, year=1900, user_agent='getweb agent', timeout=10):
    archive_url = 'https://web.archive.org/web/{}/{}'.format(year, url)
    try:
        r = requests.get(archive_url, headers={'User-Agent': user_agent}, timeout=timeout)
    except requests.exceptions.ReadTimeout as e:
        print(e)
        return '', '', False
    if archive_url == r.url: # archive not found
        return archive(url, user_agent, timeout)
    return r.text, r.url, r.ok

def archive(url, user_agent, timeout):
    try:
        archive_url = savepagenow.capture(url)
    except savepagenow.api.WaybackRuntimeError as e:
        print(e)
        return '', '', False
    try:
        r = requests.get(archive_url, headers={'User-Agent': user_agent}, timeout=timeout)
    except requests.exceptions.ReadTimeout as e:
        print(e)
        return '', '', False
    print('archive')
    return r.text, r.url, r.ok
    
def extract(html):
    try:
        article = NewsPlease.from_html(html, url=None)
    except newspaper.article.ArticleException as e:
        print(e)
        return {}
    return {
                'title': article.title,
                'maintext': article.maintext,
                'language': article.language
            }

def index_sources(wikinews_json_path, index_path, html_path, json_path):
    
    sources = []
    files = [join(wikinews_json_path, f) for f in listdir(wikinews_json_path) if isfile(join(wikinews_json_path, f)) and join(wikinews_json_path, f)[-4:] == 'json']
    for filename in files:
        with open(filename) as json_file:
            doc = json.load(json_file)
        sources += doc['sources']
    
    sources = list(set(sources))

    index_template = '{}\t{}\t{}\t{:06d}\n'
    html_template = html_path+'/{:06d}.html'
    json_template = json_path+'/{:06d}.json'
    i = 0
    with open(index_path, 'w') as f:
        for url in tqdm(sources, desc='Index sources'):
            html, archive_url, article, ok = request(url)
            f.write(index_template.format(url, archive_url, ok, i))
            with open(html_template.format(i), 'w') as f_html:
                f_html.write(html)
            with open(json_template.format(i), 'w') as f_json:
                json.dump(article, f_json, indent=4)
            i += 1
