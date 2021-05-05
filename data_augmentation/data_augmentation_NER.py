from nltk.corpus import wordnet
from collections import defaultdict
from numpy.random import binomial, choice, randint
from ..NER.utils import get_entity_tags

#all methods
#TODO: separate by pos for LwTR and synonyms?

def get_entities(labels,sents):

    ret = defaultdict(set)
    entities = [get_entity_tags(label) for label in labels]
    for i,en in enumerate(entities):
        
        key_val = [( labels[i][e[0]], tuple(sents[i][e[0]:e[1]+1])) for e in en]
        for key,val in key_val:
            
            ret[key].add(val)
    
    return {k:list(list(vi) for vi in v) for k,v in ret.items()}

class data_augmentator:

    def __init__(self,texts,labels):
        
        """
        p: probability of swapping a word
        n_per_sentence: 
            if int: number of new sentences to generate per sentence. 
            If float < 1: probability of creating a new sentence per sentence
        texts: list of list of tokens
        labels: list of list of labels
        """
        self._texts = texts
        self._labels = labels
        self._tok_labels = defaultdict(set)

        for sents,ents in zip(texts,labels):

            for token,label in zip(sents,ents):

                self._tok_labels[label].add(token)

        #for efficient sampling dont use a set, transform back to list
        self._tok_labels = {k:list(v) for k,v in self._tok_labels.items()}
        self._entities = get_entities(labels,texts)

    def _transform(self,function,p,n_per_sentence,**kwargs):
        
        augmented_toks = [None for _ in range(n_per_sentence*len(self._texts))]
        augmented_labels = [None for _ in range(n_per_sentence*len(self._texts))]
        j = 0
        for sentence,labels in zip(self._texts, self._labels):

            for _ in range(n_per_sentence):

                n = binomial(n = 1,size = len(sentence), p = p)
                augmented_toks[j], augmented_labels[j] = function(sentence,labels,n, **kwargs)
                j += 1

        return augmented_toks, augmented_labels

    def _LwTR(self,sentence,labels,n):
        return [self._tok_labels[label][randint(0, len(self._tok_labels[label]))] if change else token
                        for (label,token,change) in zip(labels,sentence,n) ], labels
    def augment_LwTR(self,p,n_per_sentence):

        """
        Label-wise token replacement
        Each token in chosen sentence has a probability p
        of being replaced by another token of the same label
        """
        return self._transform(self._LwTR,p,n_per_sentence)

    def _augment_SR(self, sentence,labels,n,hypernyms = False, hyponyms = False):

        ret_tok = []
        ret_labels = []

        for i,(token,label,n) in enumerate(zip(sentence, labels, n)):

            if not n:

                ret_tok.append(token)
                ret_labels.append(label)
                continue

            words = []
            words.extend([ s.name() for ss in wordnet.synsets(token) for s in ss.lemmas()])
            if hypernyms:

                words.extend([ l.name() for ss in wordnet.synsets(token) for s in ss.hypernyms()
                    for l in s.lemmas()])

            if hyponyms:

                words.extend([ l.name() for ss in wordnet.synsets(token) for s in ss.hyponyms()
                    for l in s.lemmas()])

            if not words:
                
                ret_tok.append(token)
                ret_labels.append(label)
                continue

            chosen = choice(words)
            tokens = chosen.split("_")
            if label[0] == "B":

                t_labels = [label] +  [ "I" + label[1:] for _ in range(len(tokens) - 1)]

            else:

                t_labels = [label for _ in range(len(tokens))]

            ret_tok.extend(tokens)
            ret_labels.extend(t_labels)

        return ret_tok, ret_labels

    def augment_SR(self,p,n_per_sentence,hypernyms = False,hyponyms = False):
        
        """
        Augment with synonym replacement and
        with hypernymns 
        TODO: take into account POS?
        """
        return self._transform(self._augment_SR,p,n_per_sentence, hypernyms = hypernyms,hyponyms = hyponyms)

    def augment_MR(self,p,n_per_sentence):

        """
        For each entity, replace it with another of the same class
        """
        augmented_toks = [None for _ in range(n_per_sentence*len(self._texts))]
        augmented_labels = [None for _ in range(n_per_sentence*len(self._texts))]
        j = 0
        for sentence,labels in zip(self._texts, self._labels):

            for _ in range(n_per_sentence):

                label = get_entity_tags(labels)
                n = binomial(n = 1,size = len(label), p = p)
                label = [l for l,t in zip(label,n) if t]

                if not label:

                    augmented_toks[j] = sentence
                    augmented_labels[j] = labels
                else:
                    
                    to_replace = [choice(self._entities[labels[l[0]]] ) for l in label]
                    to_replace_tags = [ ["B" + labels[l[0]][1:]] + [ "I" + labels[l[0]][1:] 
                        for _ in range(len(to_replace[i]) - 1) ] for i,l in enumerate(label) ]
                            
                    assert list(map(len,to_replace)) == list(map(len,to_replace_tags))
                    
                    augmented_toks[j] = []
                    augmented_labels[j] = []
                    s = 0

                    for i in range(len(label)):

                        augmented_toks[j] += sentence[s:label[i][0]] + to_replace[i]
                        augmented_labels[j] += labels[s:label[i][0]] + to_replace_tags[i]
                        s =  label[i][1] + 1

                    augmented_toks[j] += sentence[s:]
                    augmented_labels[j] += labels[s:]
                j += 1

        return augmented_toks, augmented_labels
