"""Collection scorings calculation"""
import sys
import codecs
from django.conf import settings

_EXPIRE = sys.maxint

# import POS tagging scorings
from .postag import *
# import ACM DL search scores
from .dl_acm_search import *


# ScienceWISE ontology
ontology = set(codecs.open(settings.ABS_PATH('ontology.csv'), 'r', 'utf-8').read().split('\n'))
