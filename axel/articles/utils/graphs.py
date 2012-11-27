"""Various graph utilities"""
import networkx as nx
from axel.articles.models import Article


def get_nx_collocations_graph():
    """Build collocations graph using networkx"""
    colloc_graph = nx.Graph()
    for article in Article.objects.all():
        collocs = list(article.articlecollocation_set.all())
        for i, c1 in enumerate(collocs):
            for j in range(i+1,len(collocs)):
                c2 = collocs[j]
                if colloc_graph.has_edge(c1.keywords, c2.keywords):
                    edge = colloc_graph[c1.keywords][c2.keywords]
                    if edge.has_key('weight'):
                        edge['weight'] += 1
                    else:
                        edge['weight'] = 2
                else:
                    colloc_graph.add_edge(c1.keywords, c2.keywords)
    return colloc_graph
