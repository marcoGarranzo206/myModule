from nltk.tree import ParentedTree

class specialNodeFinder:
    
    """
    Given a parse tree, find nodes that meet special condition:
        belong to some label (typically POS or a chunk)
        do not have some label inside them (typically POS or chunk)
        
    Example: find basic NP as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4464931/
    That is, find all NP that do not include a S, SBAR, VP or P
    """
    
    def __init__(self,name: str, look_for: set, avoid: set):
        
        self.name = name
        self.look_for = look_for
        self.avoid = avoid
        
    def relabel(self, tree: ParentedTree, overwrite = True, sep = "_"):
        
        self.special_labels = set()
        self._find_special_nodes(tree)
        
        for k in self.special_labels:
            
            if overwrite:
                
                tree[k]._label = self.name
                
            else:
                
                
                tree[k]._label += sep + self.name
                
    def find_biggest(self,tree, overwrite = True, sep = "_"):
        
        """
        Relabels the tree and returns all special nodes 
        that do not have a special node in them
        """
        
        self.relabel(tree, overwrite = True, sep = "_")
        visited = set()
        def _DFS(tree, inside = False):

            if not type(tree) is str:

                #print(tree.treeposition())
                if tree.treeposition() in self.special_labels:
                    
                    if inside:
                        
                        self.special_labels.remove(tree.treeposition())
                        
                    else:
                        
                        inside = True
                        
                for node in tree:

                    if id(node) not in visited:

                        visited.add(id(node))
                        _DFS(node,inside)

        _DFS(tree,False)
        return self.special_labels
            
        
    def _find_special_nodes(self, node: ParentedTree):

        #IMP assumption: dealing with a tree
        #as such dont have to keep track of visited nodes
        #since each node has only one parent
       
        if not type(node) is str:

            #print(node.treeposition())
            if (node.label() not in self.look_for) and (node.label() not in self.avoid):

                res =  {self._find_special_nodes(n) for n in node }
                if 0 in res:

                    return 0

                else:

                    return 2

            elif node.label() in self.avoid:

                {self._find_special_nodes(n) for n in node }
                return 0

            elif node.label() in self.look_for:

                res =  {self._find_special_nodes(n) for n in node }
                if 0 in res:

                    return 0

                else:

                    self.special_labels.add(node.treeposition())
                    return 1
        else:

            return 2
