"""
Slight modification of Tree class of tree_lstm package
to work with GRU cells as well as a tree classifier
(One class per tree) and GRU trees and cells classes
"""

from tree_lstm import *
import torch
import dgl
from copy import deepcopy

class Tree:

    #because we added the option of a GRU cell
    #we had to modify the tree class a bit
    def __init__(self, h_size,type_ = "LSTM"):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size
        self.type = type_

    def add_node(self, parent_id=None, tensor:torch.Tensor = torch.Tensor()):

        if self.type == "LSTM":

            # is use GRU cell with this type, c tensors will just end up wasting space
            data = {'x': tensor.unsqueeze(0),
                    'h': tensor.new_zeros(size=(1, self.h_size)),
                    'c': tensor.new_zeros(size=(1, self.h_size))}

        else:

            data = {'x': tensor.unsqueeze(0),
                    'h': tensor.new_zeros(size=(1, self.h_size))}

        self.dgl_graph.add_nodes(1, data=data)
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        if parent_id:
            self.dgl_graph.add_edge(added_node_id, parent_id)
        return added_node_id

    def add_node_bottom_up(self, child_ids, tensor: torch.Tensor):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        for child_id in child_ids:
            self.dgl_graph.add_edge(child_id, added_node_id)
        return added_node_id

    def add_link(self, child_id, parent_id):
        self.dgl_graph.add_edge(child_id, parent_id)


class tree_classifier(torch.nn.Module):
    
    def __init__(self, x_size, h_state, dropout, rnn_type = "LSTM" ,cell_type = "child_sum",n_ary = None, n_classes = 2):
        

        super(tree_classifier, self).__init__()
        
        if rnn_type == "LSTM":
            
            self.tree = TreeLSTM(x_size=x_size, h_size=h_state, dropout=dropout, \
                             cell_type=cell_type, n_ary=n_ary, num_stacks = 1)
        else:
            
            #print("Hello")
            self.tree = TreeGRU(x_size=x_size, h_size=h_state, dropout=dropout, \
                 cell_type=cell_type, n_ary=n_ary, num_stacks = 1)
                
        self.fc = torch.nn.Linear(h_state,n_classes)
    def forward(self,trees,root_indexes):
        
        batched_tree = BatchedTree(trees)
        res = self.tree(batched_tree)
        
        root_nodes = res.get_hidden_state()
        root_nodes = torch.stack([root_nodes[i,x,:] for i,x in enumerate(root_indexes)])
        return self.fc(root_nodes)


class TreeGRU(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 cell_type='n_ary',
                 n_ary=None,
                 num_stacks=2):
        super(TreeGRU, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        if cell_type == 'n_ary':
            self.cell = NaryTreeGRUCell(n_ary, x_size, h_size)
        else:
            self.cell = ChildSumTreeGRUCell(x_size, h_size)
        self.num_stacks = num_stacks

    def forward(self, batch: BatchedTree):
        batches = [deepcopy(batch) for _ in range(self.num_stacks)]
        for stack in range(self.num_stacks):
            cur_batch = batches[stack]
            if stack > 0:
                prev_batch = batches[stack - 1]
                cur_batch.batch_dgl_graph.ndata['x'] = prev_batch.batch_dgl_graph.ndata['h']
            cur_batch.batch_dgl_graph.register_message_func(self.cell.message_func)
            cur_batch.batch_dgl_graph.register_reduce_func(self.cell.reduce_func)
            cur_batch.batch_dgl_graph.register_apply_node_func(self.cell.apply_node_func)
            cur_batch.batch_dgl_graph.ndata['ruo'] = self.cell.W_ruo(self.dropout(batch.batch_dgl_graph.ndata['x']))
            dgl.prop_nodes_topo(cur_batch.batch_dgl_graph)
        return batches


class ChildSumTreeGRUCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeGRUCell, self).__init__()
        self.W_ruo = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_ruo = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_ruo = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        return {'ruo': nodes.data['ruo'] + self.U_ruo(h_tild), "h": h_tild}

    def apply_node_func(self, nodes):
        ruo = nodes.data['ruo'] + self.b_ruo
        r, u, o = torch.chunk(ruo, 3, 1)
        r, u, o = torch.sigmoid(r), torch.sigmoid(u), torch.tanh(o)
        u_2 = (1-u)*nodes.data["h"]
        h = o * u + u_2
        return {'h': h}
    
class NaryTreeGRUCell(torch.nn.Module):
    def __init__(self, n_ary, x_size, h_size):
        super(NaryTreeGRUCell, self).__init__()
        self.n_ary = n_ary
        self.h_size = h_size
        self.W_ruo = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_ruo = torch.nn.Linear(n_ary * h_size, 3 * h_size, bias=False)
        self.b_ruo = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        #need another matrix to multiply the hidden states for u_2 calculations
        self.U_u2 = torch.nn.Linear(n_ary * h_size, h_size, bias=False)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        padding_hs = self.n_ary - nodes.mailbox['h'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)

        return {'ruo': nodes.data['ruo'] + self.U_ruo(h_cat), 'h': self.U_u2(h_cat)}

    def apply_node_func(self, nodes):
        ruo = nodes.data['ruo'] + self.b_ruo
        r, u, o = torch.chunk(ruo, 3, 1)
        r, u, o = torch.sigmoid(r), torch.sigmoid(u), torch.tanh(o)
        u_2 = (1-u)*nodes.data["h"]
        h = o * u + u_2
        return {'h': h}

