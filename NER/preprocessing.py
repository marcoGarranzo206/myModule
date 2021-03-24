from nltk import pos_tag
from nltk.stem import snowball, WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = snowball.EnglishStemmer()
import re
import sys
from myModule import extraPrograms
from myModule.objects import bidict
import numpy as np

nltk_pos = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS', '#']
pos2idx = bidict({pos:i for (i,pos) in enumerate(nltk_pos)})

bioLemma = r"/media/marco/TOSHIBA\ EXT/programs/NLP/biolemmatizer-1.2/"


def Case(sentences):
    
    return [[[token.islower(), token.isupper(), token.istitle()] for token in sentence] for sentence in sentences]

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def nltk_lemmatizer(sentences):
    
    return [[lemmatizer.lemmatize(token, get_wordnet_pos(token_pos)) for token,token_pos in token_pos_list] \
            for token_pos_list in sentences ]

def bio_lemmatize(sentences):

    lemmatized_tokens = extraPrograms.bioLemmatizer(sentences, bioLemma)
    return [[token[0] for token in sentence ] for sentence in lemmatized_tokens ]
    
def stem(sentences):
    
    return [[stemmer.stem(token) for token in sentence] for sentence in sentences] 


def hasSuffix(token,suffix_list):
    
    one_hot_encoder = [0 for _ in suffix_list]
    
    for i, suffix in enumerate(suffix_list):

        one_hot_encoder[i] = token.endswith(suffix)
        
    return one_hot_encoder

def Suffix(sentences, suffix_list):
    
    return [[hasSuffix(token, suffix_list) for token in sentence] for sentence in sentences ]
    
def hasPrefix(token,prefix_list):
    
    one_hot_encoder = [0 for _ in prefix_list]
    
    for i, prefix in enumerate(prefix_list):

        one_hot_encoder[i] = token.startswith(prefix)
        
    return one_hot_encoder

def Prefix(sentences, prefix_list):
    
    return [[hasPrefix(token, prefix_list) for token in sentence] for sentence in sentences ]

def retain_alpha_numeric(token):
    
    return re.sub('[^A-Za-z0-9]+', '', token)

def pos_one_hot(sentences):
    
    to_ret = [[[0 for _ in range(len(pos2idx))] for _ in sentence] for sentence in sentences]
    for i,sentence in enumerate(sentences):
        
        for j,pos in enumerate(sentence):
            
            to_ret[i][j][pos2idx[pos]] = 1
            
    return to_ret

def inGazetteerToken(token, gazetter_list):
    
    return [ token in gazetteer for gazetteer in gazetter_list]

def inGazetteer(sentences,gazetter_list):
    
    return [ [inGazetteerToken(token,gazetter_list) for token in sentence] for sentence in sentences]


class tokenizedSentenceFeatureExtraction:
    
    """
    Feature extractor for already tokenized sentences
    It is assumed that the tokens you have are what you want to give to the model
    although we allow for stemming/lemmatization/normalization in order to extract features
    that might otherwise be lost with them, such as suffixes, or casing info

    presence in gazetteers is done on normalized token + lemmatized, so be sure to normalize them beforehand

    what cannot happen is for tokens to be expanded or removed (try at your own risk)
    ie no removing tokens because they are stop words/have only special chars (should ALREADY BE DONE)
    ie if using bert, we are using already the split tokens
    
    to allow for flexibility in processing, this object works by passing functions which should extract
    relevant features
    
    all functions passed must  take as input a list of list of tokens, output a list of list of token attributes
    
    given a list of sentences
    
        for each sentence
            for each word
                extract_features(word)
                
        return list of list of features. As many lists as sentences. In each list, as many lists as
        tokens with their features extracted
        
    allowed features:
    
        case information: return as many boolean variables as case info you may want
        affix info: return a one hot encoded vector. Each dimension correponds to an affix.
        True if word contains affix
        embedding: extract word vector 
        gazetter: vector of boolean variables, one per gazzetteer. returns true if token is in 
        specified gazetteer
        POS tag
        extra_features

    
    """
    
    def __init__(self, normalizer, lemma, prefixes, suffixes,\
                gazetteers,embedding):

        
        self.lemma = lemma
        self.embedding = embedding
        self.normalizer = normalizer
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.gazetteers = gazetteers

        
    def transform(self, X):
        
        to_ret = [None for _ in X]
        sentences_norm = self.normalizer(X)
        sentences_pos =  [pos_tag(sentence) for sentence in sentences_norm]
        POS = [[pos for _,pos in sentence] for sentence in sentences_pos]
        
        one_hot_pos = pos_one_hot(POS)
        X_lemma = self.lemma(sentences_pos)
        case_info = Case(X)
        prefix_info = Prefix(X, self.prefixes)
        suffix_info = Suffix(X, self.suffixes)
        embedding_info = self.embedding(X_lemma)
        gazetteers = inGazetteer(X_lemma, self.gazetteers)
        
        for i in range(len(X)):
            

            as_array = (np.array([po+c+pr+su+ga+e for (po,c,pr,su,ga,e) \
                            in zip(one_hot_pos[i], case_info[i],prefix_info[i], suffix_info[i],\
                                   gazetteers[i],embedding_info[i])]))            
            to_ret[i] = as_array
    
        return to_ret

def window(features,window_length):
    
    """
    given a sequence of features, each with the same elements
    extract for element in the sequence, the features of elements
    window size to the left, its features, features of elements window size the right
    
    for elements at ends of sequence, extract all zeros
    
    given e1,e2,e3,e4 and window_length = 1
    
    extract:
    
        0,e1,e2
        e1,e2,e3
        e2,e3,e4
        e3,e4,0
        
    where 0 is the 0th vector, ei is the vector of ith element
    """
    
    if window_length == 0:
        
        return features
    
    n_elements = features.shape[1]
    n_per_feature = 2*window_length+1
    to_ret = [None for _ in range(len(features))]
    for i in range(len(features)):
        
        curr_feature = features[i]
        to_ret[i] = np.zeros((n_per_feature,n_elements))
        
        for idx,j in enumerate(range(i-window_length,i)):
        
            if j < 0:
                
                pass
                
            else:
                
                to_ret[i][idx] = features[j]
        
        idx += 1
        to_ret[i][idx] = curr_feature
        idx += 1
        for idx,j in enumerate(range(i+ 1, i + window_length + 1), start = idx):
        
            if j >  len(features) - 1:
                
                pass
                
            else:
                
                to_ret[i][idx] = features[j]
                
    return to_ret
