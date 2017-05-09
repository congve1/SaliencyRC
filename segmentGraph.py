import disjointSet
class edge:
    def __init__(self,w=0,a=0,b=0):
        self.w = w
        self.a = a
        self.b = b


def threshold(size,c):
    return c / size

"""
Segment a graph
Returns a dis
"""