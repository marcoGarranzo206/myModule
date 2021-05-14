import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import sys
import gensim
from .utils import check_sentence_lengths
from sklearn.metrics import f1_score
import numpy as np

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
        
        """
        num_embeddings: number of different ids. Padding and unk assumed to not be included
        embedding_dim: size of the embedding for the ids
        hidden_size: hidden size of bilstm layer. The returned vector will be 2*this number, since
        the last cell is returned in forward and reverse
        num_layers: number of layers for bilstm layer
        """

        super(BiLSTM_embeddings, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = num_embeddings
        self.embedding = nn.Embedding(num_embeddings+2, embedding_dim, padding_idx= num_embeddings,\
                                      **embedding_kwargs).double()
        self.bilstm = nn.LSTM(embedding_dim,hidden_size,num_layers,**bilstm_kwargs, bidirectional = True).double()
        
    def forward(self,idxs,train = True):
        
        idxs = pad_sequence(idxs,batch_first=True, padding_value=self.padding_idx)

        embedded = self.embedding(idxs)
        batch,seq_len,input_size = embedded.shape
        #embedded = embedded.reshape(seq_len, batch, input_size)
        output, (hidden, cell_state)  = self.bilstm(embedded)       
    
        return output[:,-1,:]

class char_BiLSTM(nn.Module):

    """
    bilstm with crf output layer with character embedding layer, token embedding layer
    and optionally extra features that can be passed (ie POS tags)

    Designed for NER in mind, but can be useful for any seq2seq text classification task.
    Given a sentence, or list of tokens, feed into the model,for each token:

        a vector representation based on the character embedding network
        a token vector. Can use any pretrained embeddings in word2vec format or from scratcth. Can be retrained
        extra features for that token

    The output is logsoftmax for each token and class, which is passed to crf layer.
        
    """
    def __init__(self,
                 num_classes,
                 num_embeddings_chars,
                 embedding_dim_chars,
                 num_layers_chars,
                 hidden_size_tokens,
                 num_layers, 
                 vocab,
                 char_vocab,
                 extra_token_attributes = 0,
                 hidden_size_chars = None,
                 embedding_dim_tokens = None,
                 embedding_dim_path = None,
                 embedding_kwargs = {},
                 bilstm_kwargs = {},
                 freeze = False,
                dropout = 0.5):
        
        """
        num_classes: number of classes to predict
        num_embeddings_chars: number of unique characters. pad and unk not included
        embedding_dim_chars: number of embedding dimensions for character embedder layer
        num_layers_chars: number of layers for bilstm in character embedder module
        hidden_size_tokens: hidden size of bilstm layer in token embedder module
        num_layers: number of layerss for bilstm in token + extra attributes + character module
        vocab: mapper class object. Maps tokens to indices for the token embedding
        char_vocab: mapper class object. Maps characters to indices for character level token embedding
        extra_token_attributes: number of extra features to include for each token aside from embedding and character level emebedding
        hidden_size_chars: hidden size of bilstm layer in character embedder module. If none, set to number of token embedding dimensions / 2
        embedding_dim_tokens: number of dimensions for token embedding. If using pretrained, doesnt do anything
        embedding_dim_path: path to pretarined word2vec
        freeze: when using pretrained token embeddings, whether to freeze the or not
        dropout: dropout probability for bilstm in token + extra attributes + character module layers plus after that layer
        """

        super(char_BiLSTM, self).__init__()
        if embedding_dim_path is not None:
            
            word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_dim_path)
            word_vectors.add("<PAD>", weights = np.random.randn(word_vectors.vector_size) )
            word_vectors.add("<UNK>", weights = np.random.randn(word_vectors.vector_size) )
            embedding_dim_tokens = word_vectors.vector_size            
            self.token_embedding = nn.Embedding.from_pretrained(torch.tensor(word_vectors.vectors).double(),
                                                               padding_idx=embedding_dim_tokens,
                                                                   freeze = freeze).double()
            
        else: 
            
            self.token_embedding = nn.Embedding(num_embeddings = len(vocab), 
                                                embedding_dim = embedding_dim_tokens+2,
                                               padding_idx= embedding_dim).double()

        if hidden_size_chars is None:
            hidden_size_chars = embedding_dim_tokens//2
            
        #character layer
        self.vocab = vocab
        self.char_vocab = char_vocab
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

        self.extra_token_attributes = extra_token_attributes
        self.lstm = nn.LSTM(embedding_dim_tokens + extra_token_attributes + hidden_size_chars*2,
                            hidden_size_tokens,
                            num_layers,
                            batch_first = True,
                            dropout = dropout,
                           bidirectional = True).double()

        self.fc = nn.Linear(hidden_size_tokens*2, num_classes).double()
        self.dropout = nn.Dropout(dropout).double()
        self.crf = CRF(num_tags=num_classes, batch_first=True).double()

    def forward(self,x, extra_features = None, train = True):

        """
        At the moment: returns transition prob for crf layer
        
        crf layer: forward computes loss already
        decode can hard classify 
        
        return putput of fc or logsoftmax?
        """
        token_idxs = [ torch.tensor(self.vocab[sent]) for sent in x]
        token_idxs = pad_sequence(token_idxs, batch_first=True, padding_value = self.token_embedding.padding_idx)
        token_embedded = self.token_embedding(token_idxs)
        if extra_features:
            
            padded_extra_features = pad_sequence(extra_features, batch_first=True,
                                                 padding_value = self.extra_token_attributes)
            token_embedded_padded = torch.cat((token_embedded.double(),padded_extra_features.double()),2)
        
        
        #mask: for crf        
        char_ids = [[torch.tensor(self.char_vocab[t]) for t in sent] for sent in x ]    
        
        length_sents = list(map(len,char_ids))
        max_length = max(length_sents)
        mask = torch.tensor([ [1]*length + [0]*(max_length- length) for length in length_sents  ], dtype=bool)
        
        padded_chars = pad_sequence([self.char_embedder(idxs) for idxs in char_ids], batch_first=True)
        catted = torch.cat([token_embedded_padded.double(),padded_chars.double()], 2).double()

        out1, (hidden, cell_state) = self.lstm(catted)
        out1 = self.dropout(out1)

        out2 = self.fc(out1)
        out3 = torch.log_softmax(out2, dim = 1)
        return out3, mask

    def predict(self,x, extra_features = None):

        out, mask = self.forward(x, extra_features, train = False)
        return self.crf.decode(out, mask)
    
    def fit(self,
              train_sents,
              train_extra,
              valid_sents,
              valid_extra,
              Y_train,
              Y_valid,
              n_epochs,
              lr,
              weight_decay,
              gradient_clipping,
             batch_size,
             betas = (0.9,0.999),
             eps = 1e-08,
             amsgrad = False):
        
        check_sentence_lengths(train_sents, train_extra, "train sentences and train features")
        check_sentence_lengths(train_sents, Y_train, "train sentences and train tags")
        
        check_sentence_lengths(valid_sents, valid_extra, "valid sentences and valid features")
        check_sentence_lengths(valid_sents, Y_valid, "valid sentences and valid tags")
        
        #make batches with similar sized sentences. Avoid
        #exesive padding when putting short with long sentences
        idxs = np.argsort(list(map(len,train_sents ))) 
        
        #TODO: let optimizer be passed, along with its **kwargs?
        optimizer = optim.Adam(self.parameters(), 
                               lr = lr,
                               weight_decay = weight_decay,
                               betas = betas,
                               eps = eps,
                              amsgrad = amsgrad)
        
        #TODO:early stopping after validation score doesnt increase/decreases?
        for i in range(n_epochs):

            self.train()
            for batch in range(len(train_sents)//batch_size + 1):

                print(".", end="")
                batch_idx = idxs[batch*batch_size: batch_size*(1+batch)]
                sentences = [train_sents[i] for i in batch_idx]
                extra = [train_extra[i] for i in batch_idx]
                outputs = pad_sequence([torch.tensor(Y_train[i]) for i in batch_idx], 
                                       batch_first=True)

                out,mask = self.forward(sentences,extra)
                loss = - self.crf(out,outputs,mask)
                loss.backward()       

                torch.nn.utils.clip_grad_norm_(self.parameters(), 
                                               gradient_clipping, 2)
                optimizer.step()
                optimizer.zero_grad() 

            print()
            yhat_train = []
            yhat_valid = []
            self.eval()

            for sents,features in zip(train_sents,train_extra):

                yhat_train.extend( self.predict([sents],[features])[0] )

            for sents,features in zip(valid_sents,valid_extra):

                yhat_valid.extend( self.predict([sents],[features])[0] )

            #TODO: custom evaluation score? receive unfolded list of entities?unroll once and save it as a variable
            macro_train = f1_score([t for sent in Y_train for t in sent],yhat_train, average="macro" )
            micro_train = f1_score([t for sent in Y_train for t in sent],yhat_train, average="micro" )

            macro_valid = f1_score([t for sent in Y_valid for t in sent],yhat_valid, average="macro" )
            micro_valid = f1_score([t for sent in Y_valid for t in sent],yhat_valid, average="micro" )
            
            print(f"epoch: {i + 1}")
            print(f"f1 macro train: {macro_train}")
            print(f"f1 micro train: {micro_train}")
            print(f"f1 macro valid: {macro_valid}")
            print(f"f1 micro valid: {micro_valid}")
            
        return macro_valid, micro_valid
