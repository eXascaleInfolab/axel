"""Match extracted collocation with DBPedia entities"""

from datetime import date
from lxml import etree
import requests
from suds.client import Client
from suds.xsd.doctor import Import, ImportDoctor


DBLP_URL = 'http://dblp.l3s.de/WS/aspl2.php?wsdl'
imp = Import('http://schemas.xmlsoap.org/soap/encoding/',
             location='http://schemas.xmlsoap.org/soap/encoding/')
dblp_client = Client(DBLP_URL,plugins=[ImportDoctor(imp)])

DBPEDIA_REQ = u'http://lookup.dbpedia.org/api/search' \
              u'.asmx/KeywordSearch?QueryClass=&QueryString={0}&MaxHits=1'


def perform_match(collocation):
    """
    :type collocation: Collocation
    """
    source = []
    # perform search using dbpedia
    r = requests.get(DBPEDIA_REQ.format(collocation.ngram))
    xml = etree.fromstring(r.text.replace('encoding="utf-8"', ''))
    result = xml.find('.//{http://lookup.dbpedia.org/}Label')
    desc = xml.find('.//{http://lookup.dbpedia.org/}Description')
    if result is not None and desc is not None and result.text.lower() == collocation.ngram:
        source.append("dbpedia")

    # perform keyword search from dblp
    dblp_res = dblp_client.service.all_keywords_year(searchTerm=collocation.ngram, limit=1,
                                                     startYear=1999, endYear=date.today().year)
    if dblp_res and dblp_res[0].keyword.lower() == collocation.ngram:
        source.append("dblp")

    return source
