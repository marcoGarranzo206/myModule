import sys
sys.path.append("/home/marco/")
from myModule.NER import loader
from myModule.NER.preprocessing import tokenizedSentenceFeatureExtraction as tSFE  
from myModule.NER.preprocessing import window
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from myModule import extraPrograms 
from collections import Counter
from gensim.models.keyedvectors import Word2VecKeyedVectors
import gensim
import numpy as np

def sent_tokenize_with_newlines(text):
    
    line_break_splitted = text.split("\n")    
    return [sent for line in line_break_splitted for sent in sent_tokenize(line)]

class word2vec_embedder:

    def __init__(self, word2vec):

        self.word2vec = word2vec

    def __call__(self, sentences):
        
        #print(X)
        tokens_embedding = [ [None for token in sentence] for sentence in sentences ]
        for i,sentence in enumerate(sentences):
            
            for j,token in enumerate(sentence):
            
                try:

                    tokens_embedding[i][j] = [vi for vi in self.word2vec[token]]

                except KeyError:

                    tokens_embedding[i][j] = [vi for vi in self.word2vec["UNK"]]
                
        return tokens_embedding

def no_process(x):

    return x

def load_DDI_Corpus(text_processor,
        word_tokenizer,
        normalizer,
        embedding_type,
        embedding_direc,
        gazetteers,
        prefix_function,
        suffix_function):

    root = "/media/marco/TOSHIBA EXT/datasets/NLP/DDI/DDICorpusBrat/Train/"
    drugBankRoot = root + "DrugBank/"
    MedLineRoot = root + "MedLine/"

    drugbankSentences, drugbankTags,_ = loader.load_DDI(drugBankRoot, no_process,sentence_tokenizer=sent_tokenize_with_newlines, \
                                                      word_tokenizer = word_tokenizer, start_index = 1)
    medlineSentences, medlineTags,_ = loader.load_DDI(MedLineRoot, no_process, sentence_tokenizer=sent_tokenize_with_newlines, \
                                                      word_tokenizer = word_tokenizer)

    rootTest = "/media/marco/TOSHIBA EXT/datasets/NLP/DDI/DDICorpusBrat/Test/"
    drugBankRootTest = rootTest + "DrugBank/"
    MedLineRootTest = rootTest + "MedLine/"

    drugbankSentencesTest, drugbankTagsTest,_ = loader.load_DDI(drugBankRootTest, no_process, sentence_tokenizer=sent_tokenize_with_newlines, \
                                                      word_tokenizer = word_tokenizer)
    medlineSentencesTest, medlineTagsTest,_ = loader.load_DDI(MedLineRootTest, no_process,sentence_tokenizer=sent_tokenize_with_newlines, \
                                                      word_tokenizer = word_tokenizer)
    assert list(map(len, drugbankSentences)) == list(map(len, drugbankTags))
    assert list(map(len, medlineSentences)) == list(map(len, medlineTags))

    assert list(map(len, drugbankSentencesTest)) == list(map(len, drugbankTagsTest))
    assert list(map(len, medlineSentencesTest)) == list(map(len, medlineTagsTest))
    print("Finished initial tokenization")

    if embedding_type == "gensim_keyedVector":

        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_direc,binary = True)
        embedding_function = word2vec_embedder(word2vec)

    elif embedding_type == "BERT":

        raise NotImplementedError("No suport for BERT tokenization just yet!")

    print("Finished loading word embedders")
    sets = [set() for _ in gazetteers]

    for file, set_ in zip(gazetteers,sets):

        with open(file, "r") as f:

            for line in f:

                set_.update([line[:-1]])

    sets = list(map(lambda x: set(normalizer([list(x)])[0]), sets))
    print("Finished loading gazetteers")
    
    all_sentences = drugbankSentences + medlineSentences
    all_tags = drugbankTags + medlineTags

    all_sentences_test = drugbankSentencesTest + medlineSentencesTest
    all_tags_test = drugbankTagsTest + medlineTagsTest

    prefixes = prefix_function(all_sentences)
    suffixes = suffix_function(all_sentences)
    print("Finsihed extracting affixes")

    t = tSFE(text_processor, normalizer, prefixes,suffixes,sets, embedding_function)
    X_train = t.transform(all_sentences)
    X_test = t.transform(all_sentences_test)
    
    print("Finished loading data")
    return X_train, X_test, all_tags, all_tags_test, all_sentences, all_sentences_test

