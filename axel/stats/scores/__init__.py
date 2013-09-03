"""Collection scorings calculation"""
import sys

_EXPIRE = sys.maxint

# import POS tagging scorings
from .postag import *
# import ACM DL search scores
from .dl_acm_search import *
