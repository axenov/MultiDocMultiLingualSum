import logging
import re
from smart_open import open
from xml.etree import cElementTree
import json
import pandas as pd

from gensim.corpora.wikicorpus import get_namespace, filter_wiki
from gensim.scripts.segment_wiki import extract_page_xmls

logger = logging.getLogger(__name__)

def find_sources(text, sources_translations, footnote_pattern, url_pattern):
    sources = []
    for footnote in footnote_pattern.findall(text):
        footnote_title = list(footnote)[0].replace(' ', '').lower()
        footnote_content = list(footnote)[1].split('\n*')[1:]
        if footnote_title in sources_translations:
            for raw_source in footnote_content:
                sources += url_pattern.findall(raw_source)
    return sources

def get_pages_from_wiki_dump(wiki_dump_path, max_doc_count=0):

    sources_translations = ['quellen', 'sources', 'quelle', 'source']

    category_pattern = re.compile('\[\[(Category|Kategorie):(.*?)\]\]')
    footnote_pattern = re.compile(r'==(.+?)==(.+?)\n *\n', flags=re.DOTALL)
    url_pattern = re.compile(r'https?://[^\s|\]]+')
    blank_pattern = re.compile(r'^\s*$')

    with open(wiki_dump_path, 'rb') as xml_fileobj:
        page_xmls = extract_page_xmls(xml_fileobj)
        i = 0

        docs = []

        for i, page_xml in enumerate(page_xmls):
            
            elem = cElementTree.fromstring(page_xml)
            filter_namespaces = ('0',)
            namespace = get_namespace(elem.tag)
            ns_mapping = {"ns": namespace}
            text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
            title_path = "./{%(ns)s}title" % ns_mapping
            ns_path = "./{%(ns)s}ns" % ns_mapping

            title = elem.find(title_path).text
            text = elem.find(text_path).text
            ns = elem.find(ns_path).text
            if ns not in filter_namespaces:
                continue

            try:

                categories = [c for _, c in category_pattern.findall(text)]

                sources = find_sources(text, sources_translations, footnote_pattern, url_pattern)
                if sources == [] :
                    continue
                
                cleaned_text = category_pattern.sub('', text)
                cleaned_text = footnote_pattern.sub('', cleaned_text)
                cleaned_text = filter_wiki(cleaned_text)
                passages = [passage for passage in cleaned_text.split('\n\n') if blank_pattern.match(passage) == None]

                if len(' '.join(passages).split()) == 0:
                    continue
                
                if '#REDIRECT' in cleaned_text or '#redirect' in cleaned_text:
                    continue

                docs.append({
                    'title': title,
                    'text': passages,
                    'categories': categories,
                    'sources': sources,
                })

                if 0 < max_doc_count < len(docs):
                    break
            except (TypeError, ValueError) as e:
                logger.error(f'Cannot read page #{i} - {title}: {e}')

    return docs

def stats(data_path, csv_path, save_csv):
    titles = []
    num_words = []
    num_sources = []
    with open(data_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            title = doc['title']
            text = ' '.join(doc['text'])
            sources = doc['sources']

            if len(text.split()) == 0:
                print(title)
                print(text)

            titles.append(title)
            num_words.append(len(text.split()))
            num_sources.append(len(sources))

    data = {'title': titles, 'num_words': num_words, 'num_sources': num_sources}
    df = pd.DataFrame(data=data)
    if save_csv:
        df.to_csv(csv_path, index=False)
    print(df.describe())
    
