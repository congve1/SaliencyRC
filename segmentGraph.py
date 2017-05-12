import disjointSet
class Edge:
    def __init__(self,w=0.0,a=0,b=0):
        self.w = w
        self.a = a
        self.b = b


def threshold(size,c):
    return c / size

"""
Segment a graph
Returns a disjoint-set forest representing the segmentation
nu_vertices:number of vertices in the graph
nu_edges: number of edges in the graph
c: constant for threshold function
"""
def  segment_graph(nu_vertices,nu_edges,edges,c):
    tmp = edges[:nu_edges]
    #sorted by weight
    tmp.sort(key=lambda edge: edge.w)
    edges[:nu_edges] = tmp
    #make a disjoint-set forest
    u = disjointSet.Universe(nu_vertices)
    thresholds = [threshold(1,c) for _ in range(nu_vertices)]
    loop_range= range(nu_edges)
    for i in loop_range:
        edge = edges[i]
        a = u.find(edge.a)
        b = u.find(edge.b)
        if a != b:
            if edge.w <= thresholds[a] and edge.w <= thresholds[b]:
                u.join(a,b)
                a = u.find(a)
                thresholds[a] = edge.w + threshold(u.size(a),c)
    return u
