"""middleware"""
from django.core.exceptions import MiddlewareNotUsed
from axel.articles.utils.concepts_index import build_index


class ConceptIndexMiddleware(object):
    """Middleware to load initial concept index"""
    def __init__(self):
        """build index"""
        build_index()
        raise MiddlewareNotUsed
