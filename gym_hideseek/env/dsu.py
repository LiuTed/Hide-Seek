class DSU: # Disjoint Set Union
    def __init__(self, n) -> None:
        self.parent = [-1 for _ in range(n)]
    
    def find(self, element) -> int:
        '''Return the root parent of the element'''
        if self.parent[element] == -1:
            return element
        p = self.find(self.parent[element])
        self.parent[element] = p # path compression
        return p

    def add(self, element, set) -> int:
        '''Add element and the set it belongs to to the set'''
        pe = self.find(element)
        ps = self.find(set)
        if ps == pe:
            return ps
        else:
            self.parent[pe] = ps
            return ps