import numpy as np

def get_parents(graph, node, ret = None):
    
    """
    given a node and a DAG, traverse the DAG upwards,
    picking up all parents until there are none
    """
    if ret is None:
        ret = set()
    for predecessor in graph.predecessors(n = node):

        ret.add(predecessor)
        get_parents(graph, predecessor, ret)
    
    return ret

def discretize_weighted(g,cutoff,attr = "weight"):

    to_remove = []
    for u,v,d in g.edges(data = True):

        if abs(d[attr]) < cutoff:

            to_remove.append((u,v))

        else:

            g[u][v][attr] = np.sign(d[attr])

    g.remove_edges_from(to_remove)
