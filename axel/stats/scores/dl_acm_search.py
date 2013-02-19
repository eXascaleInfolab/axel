from lxml import html
import requests
import re


COUNT_RE = re.compile(r'Results 1 - 20 of ([\d,]+)')


def acm_search_result_count(ngram):
    """Get search result count from ACM Digital Library"""
    headers = {'User-Agent': "Mozilla/6.0 (Windows NT 6.2; WOW64; rv:16.0.1) Gecko/20121011 Firefox/16.0.1"}
    lxml_content = html.fromstring(requests.get('http://dl.acm.org', headers=headers).content)
    # extract search form action
    search_form_url = lxml_content.forms[0].action
    payload = {'query': '"{0}"'.format(ngram)}
    result = requests.post('http://dl.acm.org/{0}'.format(search_form_url),
        data=payload, headers=headers).content
    return int(COUNT_RE.findall(result)[0])
