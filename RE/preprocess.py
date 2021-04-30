from nltk.tree import ParentedTree
from collections import defaultdict
from .recursive_trees import Tree
import torch
from ..objects import bidict

def load_trees(tree_file, word2vec_model,h_size, type_ = "LSTM", embedding_dim = 200):

    """
    read a file of syntax trees in string format
    formatted according to XXX

    return: list of trees of type "type_" (LSTM or GRU) with hiden size
            h_size
            list of labels for each tree
    """
    trees = []
    labels = []
    with open(tree_file,"r") as f:
        
        for i,line in enumerate(f):
        

            t = ParentedTree.fromstring(line[:-1])
            #print(t[(0,)])
            trees.append(nltk_to_dgl(t,word2vec_model,h_size, type_ = type_, embedding_dim = embedding_dim))
            labels.append(t.label().split("/")[0])

    return trees,labels

def makeEntPositMat(givenInput):
    position_embed = [[1,1,1,1,1,1,1,1,1,1],
                 [1,0,1,1,1,1,1,1,1,1],
                 [1,0,0,1,1,1,1,1,1,1],
                 [1,0,0,0,1,1,1,1,1,1],
                 [1,0,0,0,0,1,1,1,1,1],
                 [1,0,0,0,0,0,1,1,1,1],
                 [1,0,0,0,0,0,0,1,1,1],
                 [1,0,0,0,0,0,0,0,1,1],
                 [1,0,0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,0,0,1,1],
                 [0,0,0,0,0,0,0,1,1,1],
                 [0,0,0,0,0,0,1,1,1,1],
                 [0,0,0,0,0,1,1,1,1,1],
                 [0,0,0,0,1,1,1,1,1,1],
                 [0,0,0,1,1,1,1,1,1,1],
                 [0,0,1,1,1,1,1,1,1,1],
                 [0,1,1,1,1,1,1,1,1,1]]
    intInput = int(givenInput)
    return position_embed[intInput]


def node_to_tensor(node,tree,word_to_parents,vocab,unk_token = "UNK", embedding_dim = 200):

    label = tree[node].label()
    _, Fsc,Fd1,Fd2 = label.split("/")
    Fd1,Fd2 = makeEntPositMat(Fd1), makeEntPositMat(Fd2)
    Fsc = [1]*10

    if node in word_to_parents:

        word = word_to_parents[node]
        if word in vocab:

            word_embedding = vocab[word]

        else:

            word_embedding = vocab[unk_token]

    else:

        word_embedding = [0]*embedding_dim

    return torch.tensor(Fd1 + Fd2 + Fsc + [x for x in word_embedding],dtype=torch.float32)

def nltk_to_dgl(tree,vocab,h_size,unk_token = "UNK",type_ = "LSTM", embedding_dim = 200):

    """
    Given a list of ntltk trees labeled according to XXX, turn them into
    dgl trees.

    Input consists of:
        word_vector: 0 if not leaf, else use vocab
        pos_info_drug_1
        pos_info_drug_2
        contextual_info: 1 or 0 if one of the drugs is in the tree (paper uses vector of ones and zeros)
    """

    dgl_tree = Tree(h_size,type_ = type_)
    word_to_parents = bidict({ tuple(list(tree.leaf_treeposition(i))[:-1]): l  for i, l in enumerate(tree.leaves())})

    edge_list = defaultdict(list)
    non_leaf = set()

    for node in tree.treepositions():

        if type(tree[node]) is str:

            continue

        non_leaf.add(node)
        for descendant in tree[node]:

            if type(descendant) is str:

                continue

            edge_list[node].append(descendant.treeposition())

    node_to_idx = dict()
    for node in non_leaf:

        tensor = node_to_tensor(node,tree,word_to_parents,vocab,unk_token, embedding_dim)
        #print(tensor.shape)
        node_to_idx[node] = dgl_tree.add_node(parent_id= None, tensor=tensor)

    for u,neighbors in edge_list.items():

        for v in neighbors:

            dgl_tree.add_link(node_to_idx[v],node_to_idx[u])
    return dgl_tree, node_to_idx[()] #hopefully root...
