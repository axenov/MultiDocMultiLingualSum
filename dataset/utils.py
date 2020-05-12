from os import listdir
from os.path import isfile, join

import json
import requests
from tqdm import tqdm

def index_sources(json_path, index_path):
    sources = []
    files = [join(json_path, f) for f in listdir(json_path) if isfile(join(json_path, f)) and join(json_path, f)[-4:] == 'json']
    for filename in files:
        with open(filename) as json_file:
            doc = json.load(json_file)
        sources += doc['sources']
    
    sources = list(set(sources))

    wayback_sources = []
    num_none = 0
    for source in tqdm(sources, desc='Get wayback sources'):
        wayback_source = get_wayback_url(source)
        if wayback_source == None:
            num_none += 1
            wayback_sources.append(source)
        wayback_sources.append(wayback_source)

    print('There are {} urls on {} which has not wayback url'.format(num_none, len(wayback_sources)))
    index_template = '{}\t{:06d}\n'
    with open(index_path, 'w') as f:
        for i, source in enumerate(wayback_sources):
            f.write(index_template.format(source, i))
    

def get_wayback_url(url):
    api_template = 'http://archive.org/wayback/available?url={}'
    response = requests.get(api_template.format(url))
    data = response.json()
    if data['archived_snapshots'] == {}:
        return None
    return data['archived_snapshots']['closest']['url']

