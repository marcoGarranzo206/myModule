import nltk
import numpy as np

def unravel_window(sents, window = 0):

    """
    Sents: list of lists (sentences) of list of token features
    returns: a list of token features, an N*M matrix where N rows
    correspond to a token and M cols correspond to each tokens attributes

    Each token takes attributes from itself and from n window tokens to the left 
    and right of it. If there are none, these features are replaced with zeros

    These tokens can then be used in classical tabular machine learning algorithms
    """

    total_tokens = sum(map(len,sents))
    n_features = len(sents[0][0])
    n_total_features = (1 + 2*window)*n_features
    ret = np.zeros((total_tokens,n_total_features))
    idx = 0
    
    for sent_i, sent in enumerate(sents):

        l = len(sent)
        for i in range(len(sent)):

            for k,j in enumerate(range(i - window, window + i + 1)):

                if j < 0 or j >= l:

                    continue
                    
                ret[idx][k*n_features:(k+1)*n_features ] = sents[sent_i][j]
                
            idx += 1
            
    return ret

def one_hot(i,tot):
    
    ret = [0 for _ in range(tot)]
    ret[i] = 1
    return ret

class pos_extractor:
    
    def __init__(self, pos = "full"):
        
        nltk_pos = ['LS', 'TO', 'VBN', "''", 
                   'WP', 'UH', 'VBG', 'JJ', 
                   'VBZ', '--', 'VBP', 'NN', 
                   'DT', 'PRP', ':', 'WP$', 
                   'NNPS', 'PRP$', 'WDT', '(', 
                   ')', '.', ',', '``', '$', 
                   'RB', 'RBR', 'RBS', 'VBD', 
                   'IN', 'FW', 'RP', 'JJR', 
                   'JJS', 'PDT', 'MD', 'VB', 
                   'WRB', 'NNP', 'EX', 'NNS', 
                   'SYM', 'CC', 'CD', 'POS', '#']
        
        if pos == "full":
            
            self.idx = max(map(len, nltk_pos))
            self.nltkpos2idx = {t:i for i,t in enumerate(nltk_pos)}
            
        else:
            
            nltk_pos = set([t[0] for t in nltk_pos])
            self.idx = 1
            self.nltkpos2idx = {t:i for i,t in enumerate(nltk_pos)}
            
        self.nltkpos2idx["UNK"] =  len(nltk_pos) + 1
        self.features = self.nltkpos2idx
        self._name = "nltkpos_extractor"
    
    def __call__(self,sents):
        
        ret = [None for _ in range(len(sents))]
        
        for i,sent in enumerate(sents):
            
            pos = [self.nltkpos2idx.get(t[1][:self.idx], len(self.nltkpos2idx)) for t in nltk.pos_tag(sent)]
            ret[i] = [one_hot(p,len(self.nltkpos2idx) + 1) for p in pos]
        return ret

class dictionary_extractor:
    
    """
    Given a word and a list of path to a file of names,
    find if that word is in that set
    example: if you want to detect drug names,
    check if its in a drug ontology
    """
    
    def __init__(self, set_list, set_names = None):
        
        self.sets = [None for _ in range(len(set_list))]
        for i, s in enumerate(set_list):
            
            self.sets[i] = self._read_file(s)
            
        if set_names is None:
            
            self.features = {path:i for i,path in enumerate(set_list)}
            
        else:
            
            self.features = {path:i for i,path in enumerate(set_names)}
            
        self._name = "dictionary_extractor"
    
    def _read_file(self, file):
        
        ret = set()
        with open(file, "r") as f:
            
            for line in f:
                
                l = line[:-1]
                ret.add(l)
                
        return ret
    
    def __call__(self, sents):
        
        return [[[token in s for s in self.sets] for token in sent] for sent in sents]
    
class extractor:
    
    def __init__(self, extractor_objects, extractor_kwargs):
        
        self.extractors = extractor_objects
        for i,kwargs in enumerate(extractor_kwargs):
            
            self.extractors[i] = self.extractors[i](**kwargs)
            
        self.features = {}
        i = 0
        for e in self.extractors:
            
            max_v = 0
            for f,v in e.features.items():
                
                self.features[e._name + "_" + f] = v + i
                if v > max_v:
                    
                    max_v = v
            i += max_v + 1
            
    def __call__(self,sents):
        
        extractor_ret = [e(sents) for e in self.extractors]
        ret = [[ [] for token in sent] for sent in sents]
        for i in range(len(sents)):
            
            for j in range(len(sents[i])):
                
                for e in extractor_ret:
                    
                    ret[i][j] += e[i][j]
                
            
        return ret
