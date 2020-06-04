from os import listdir
from os.path import isfile, join
from os import remove as rm
import os

import json
import requests
import re
import savepagenow
from tqdm import tqdm

from newsplease import NewsPlease
import newspaper

import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

import logging

counter = 0


def init_logger(restart):
    globals()["formatter"] = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    globals()["logger_wd"] = setup_logger("work_done", "work_done.log", restart)
    globals()["logger_e"] = setup_logger("error", "error.log", restart)
    globals()["logger_ok"] = setup_logger("ok", "ok.log", restart)


def setup_logger(name, log_file, restart, level=logging.INFO):
    if restart:
        os.remove(log_file)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_infos(url):
    html, archive_url, ok = get_archive(url)
    if ok:
        article = extract(html, url)
    else:
        article = {}
    return html, archive_url, article, ok


def get_archive(url, year=1900, user_agent="getweb agent", timeout=10):
    archive_url = "https://web.archive.org/web/{}/{}".format(year, url)
    try:
        r = requests.get(
            archive_url, headers={"User-Agent": user_agent}, timeout=timeout
        )
    except requests.exceptions.RequestException as e:
        logger_e.info("{} - {}".format(url, e))
        return "", "", False
    except:
        logger_e.info("{} - {}".format(url, "Unknown error"))
        return "", "", False
    if archive_url == r.url:  # archive not found
        return archive(url, user_agent, timeout)
    return r.text, r.url, r.ok


def archive(url, user_agent, timeout):
    try:
        archive_url = savepagenow.capture(url)
    except savepagenow.api.WaybackRuntimeError as e:
        logger_e.info("{} - {}".format(url, e))
        return "", "", False
    try:
        r = requests.get(
            archive_url, headers={"User-Agent": user_agent}, timeout=timeout
        )
    except requests.exceptions.RequestException as e:
        logger_e.info("{} - {}".format(url, e))
        return "", "", False
    except:
        logger_e.info("{} - {}".format(url, "Unknown error"))
        return "", "", False
    return r.text, r.url, r.ok


def extract(html, url):
    try:
        article = NewsPlease.from_html(html, url=None)
    except newspaper.article.ArticleException as e:
        logger_e.info("{} - {}".format(url, e))
        return {}
    except ValueError as e:
        logger_e.info("{} - {}".format(url, e))
        return {}
    except:
        logger_e.info("{} - {}".format(url, "Unknown error"))
        return {}
    return {
        "title": article.title,
        "maintext": article.maintext,
        "language": article.language,
    }


def index(url, index_path, index_template, html_template, json_template):
    i = uuid.uuid4()
    html, archive_url, article, ok = get_infos(url)
    with open(index_path, "a") as f:
        f.write(index_template.format(url, archive_url, ok, i))
    with open(html_template.format(i), "w") as f_html:
        f_html.write(html)
    with open(json_template.format(i), "w") as f_json:
        json.dump(article, f_json, indent=4)
    globals()["counter"] += 1
    logger_wd.info("{}/{}".format(globals()["counter"], globals()["max_counters"]))
    logger_ok.info("{} - {}".format(ok, url))


async def index_sources(
    wikinews_json_path,
    index_path,
    html_path,
    json_path,
    max_url_count,
    max_workers,
    restart,
):

    # Init logger
    init_logger(restart)

    # Get urls of sources
    sources = []
    files = [
        join(wikinews_json_path, f)
        for f in listdir(wikinews_json_path)
        if isfile(join(wikinews_json_path, f))
        and join(wikinews_json_path, f)[-4:] == "json"
    ]
    for filename in files:
        with open(filename) as json_file:
            doc = json.load(json_file)
        sources += doc["sources"]

    # Remove urls already stored and reduce to max_url_count
    sources = list(set(sources))
    with open(index_path, "r") as f:
        sources_dones = [line.split("\t")[0] for line in f.readlines()]
    sources = [source for source in sources if source not in sources_dones]
    if max_url_count != -1:
        sources = sources[:max_url_count]

    globals()["max_counters"] = len(sources)

    # Get and store infos from urls
    index_template = "{}\t{}\t{}\t{}\n"
    html_template = html_path + "/{}.html"
    json_template = json_path + "/{}.json"
    i = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                index,
                *(
                    url,
                    index_path,
                    index_template,
                    html_template,
                    json_template,
                )  # Allows us to pass in multiple arguments to `fetch`
            )
            for url in sources
        ]
        for f in asyncio.as_completed(tasks):
            await f
