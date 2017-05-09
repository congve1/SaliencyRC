class uni_elt:
    def __init__(self,rank=0,p=0,size=0):
        self.rank= rank
        self.p = p
        self.size = size

class universe:
    def __init__(self,elements):
        self.num = elements
        self.elts = [uni_elt(rank=0,p=i,size=1) for i in range(elements)]

    def join(self,x,y):
        if(self.elts[x].rank > self.elts[y].rank):
            self.elts[y].p = x
            self.elts[x].size += self.elts[y].size
        else:
            self.elts[x].p = y
            self.elts[y].size += self.elts[x].size




