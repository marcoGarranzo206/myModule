import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF

class BiLSTM_embeddings(nn.Module):
    
    """
    Given a sequence of ids to be passed to an embedding layer,
    generate a N dimensional embedding for that sequence of ids
    
    The idea is for the sequence of ids to represent a word, to have
    each id represent a letter, and therefore to learn a vector representation
    of a word taking into account spelling
    """
    
    def __init__(self,num_embeddings, embedding_dim, 
                 hidden_size, num_layers,
                 embedding_kwargs= {}, bilstm_kwargs = {}):
        
        super(BiLSTM_embeddings, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = num_embeddings
        self.embedding = nn.Embedding(num_embeddings+1, embedding_dim, padding_idx= num_embeddings,\
                                      **embedding_kwargs)
        self.bilstm = nn.LSTM(embedding_dim,hidden_size,num_layers,**bilstm_kwargs, bidirectional = True)
        
    def forward(self,idxs,train = True):
        
        idxs = pad_sequence(idxs,batch_first=True, padding_value=self.padding_idx)

        embedded = self.embedding(idxs)
        batch,seq_len,input_size = embedded.shape
        #embedded = embedded.reshape(seq_len, batch, input_size)
        output, (hidden, cell_state)  = self.bilstm(embedded)       
    
        return output[:,-1,:]

class char_BiLSTM(nn.Module):
    
    def __init__(self,embedding_dim_tokens,hidden_size_tokens,num_layers,num_classes,
                embedding_dim_chars,hidden_size_chars,num_layers_chars,num_embeddings_chars,
                 embedding_kwargs = {},bilstm_kwargs = {},
                dropout = 0.5):
        
        super(char_BiLSTM, self).__init__()
        
        
        #character layer
        self.hidden_size_chars = hidden_size_chars
        self.num_layers_chars = num_layers_chars
        self.embedding_dim_chars = embedding_dim_chars
        self.num_embedding_chars = num_embeddings_chars
        self.char_embedder = BiLSTM_embeddings(num_embeddings_chars,
                                              embedding_dim_chars,
                                              hidden_size_chars,
                                              num_layers_chars,
                                              embedding_kwargs,
                                              bilstm_kwargs)
        
        #other features layers
        self.hidden_size_tokens = hidden_size_tokens
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(embedding_dim_tokens + hidden_size_chars*2,
                            hidden_size_tokens,
                            num_layers, 
                            batch_first = True,
                            dropout = dropout,
                           bidirectional = True)
        
        self.fc = nn.Linear(hidden_size_tokens*2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        
    def forward(self,x,char_ids, train = True):
        
        """
        At the moment: returns transition prob for crf layer
        
        crf layer: forward computes loss already
        decode can hard classify 
        
        return putput of fc or logsoftmax?
        """
        length_sents = list(map(len,char_ids))
        max_length = max(length_sents)
        mask = torch.tensor([ [1]*length + [0]*(max_length- length) for length in length_sents  ], dtype=bool)
        padded = pad_sequence(x, batch_first=True)

        padded_chars = pad_sequence([self.char_embedder(idxs) for idxs in char_ids], batch_first=True)
        catted = torch.cat([padded.double(),padded_chars.double()], 2).double()
        
        out1, (hidden, cell_state) = self.lstm(catted.type(torch.float32) )
        
        if train:
            out1 = self.dropout(out1)
            
        out2 = self.fc(out1)
        out3 = F.log_softmax(out2, dim = 1)
        return out3, mask
    
    def predict(self,x,char_ids):
        
        out, mask = self.forward(x,char_ids, train = False)
        return self.crf.decode(out, mask)
    
