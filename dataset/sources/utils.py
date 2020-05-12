from newsplease import NewsPlease

def get_article(url):
    article = NewsPlease.from_url(url)
    return article 