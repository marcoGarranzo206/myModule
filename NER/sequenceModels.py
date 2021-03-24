class BiLSTMCNN(nn.Module):
    
    def __init__(self,embedding_dim,hidden_size,num_layers,num_classes,dropout = 0.5, kernel_sizes = (2,), n_kernels = 10):
        
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_layers
        
        self.lstm = nn.LSTM(embedding_dim,hidden_size,num_layers, batch_first = True,dropout = dropout,\
                           bidirectional = True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,train = True):
        
        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        
        out1, (hidden, cell_state) = self.lstm(x, (h0,c0))
        
        if train:
            out1 = self.dropout(out1)
            
        out2 = self.fc(out1)        
        out3 = F.log_softmax(out2, dim = 1)
        
        return out3


class BiLSTMCNN_with_ebeddings(BiLSTMCNN):

        def __init__(self, vocab_size, embedding_dimension, ):

            super(BiLSTMCNN, self).__init__(embedding_dimension, output_size, kernel_size,\
                                                      nodes, n_classes = 2, dropout = 0.1)
            self.embeddings = nn.Embedding(vocab_size, embedding_dimension)

