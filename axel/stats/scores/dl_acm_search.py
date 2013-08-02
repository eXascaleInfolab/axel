from lxml import html
import urllib
import requests
import re


COUNT_RE = re.compile(r'Results 1 - \d{1,2} of ([\d,]+)')
TITLE_RE = re.compile(r'<A HREF=".*?" class="medium-text" target="_self">(.*?)</A>')


def acm_search_result_count(ngram):
    """Get search result count from ACM Digital Library"""
    headers = {'User-Agent': "Mozilla/6.0 (Windows NT 6.2; WOW64; rv:16.0.1) Gecko/20121011 Firefox/16.0.1"}
    #lxml_content = html.fromstring(requests.get('http://dl.acm.org', headers=headers).content)
    # extract search form action
    ngram = urllib.quote_plus(ngram.encode('utf-8'))
    result = requests.get('http://dl.acm.org/results.cfm?query=%22{0}%22'.format(ngram),
                          headers=headers).content
    return int(COUNT_RE.findall(result)[0].replace(',', ''))


def acm_search_result_title(title):
    """Get search result count from ACM Digital Library"""
    headers = {'User-Agent': "Mozilla/6.0 (Windows NT 6.2; WOW64; rv:16.0.1) Gecko/20121011 Firefox/16.0.1"}
    #lxml_content = html.fromstring(requests.get('http://dl.acm.org', headers=headers).content)
    # extract search form action
    ngram = urllib.quote_plus(title.encode('utf-8'))
    result = requests.get('http://dl.acm.org/results.cfm?query=%22{0}%22'.format(ngram),
                          headers=headers).content
    return TITLE_RE.findall(result)[0]
