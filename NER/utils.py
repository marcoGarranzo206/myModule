import re
from ..objects import bidict
import numpy as np

def check_sentence_lengths(X,Y, name = ""):
    
    """
    sanity check after any processing at the token level
    checks whether a list of sentences (list of list of tokens)
    match in dimensions: ie same number of sentences, same sentence length
    for all sentences
    
    Importan for ex in seq2seq tasks: make sure each training token
    has a label
    """
    if len(X) != len(Y):
        
        raise ValueError(f"Different sentence length in {name}")
        
    lengths = np.array(list(map(len,X))) != np.array(list(map(len,Y)))
    
    if np.sum(lengths) != 0:
        
        w = np.where(lengths)
        raise ValueError(f"Sentence lengths do not match in {name}: different in {w}")

def anonymize(text,spans, anonymize):

    """
    Anonymize entities in a text. 
    Replaced by anonymize parameter plus an integer
    to distinguish the anonymized enitites in the text 
    and to be able to map back to them.
    
    text: text to anonymize
    span: locations in text corresponding to an entitiy
        (start,end,class,normalized_name)
    anonymize: what string to replace the word with

    same entities (different spans corresponding to the same sequence of characters)
    are mapped to the same anonymized entitiy

    Returns anonymized text and a mapping of anonymous to original entities
    """
    anonymized = []
    prev_s = 0
    ent_dict = bidict()
    i = 0
    
    for span in spans:
        
        s,e= span[0], span[1]
        name = span[3]
        
        if name not in ent_dict:
            
            ent_dict[name] = anonymize + str(i)
            i += 1
            
        anonymized.append(text[prev_s:s])
        anonymized.append(ent_dict[name])
        prev_s = e
        
    anonymized.append(text[prev_s:])
    return "".join(anonymized), ent_dict


def keep_longest_overlaps(spans,tiebreaker = None):

    
    """
    Given a list of two, or optionally three element tuples, where the first index
    is the start position and the second index is the end position
    of some element in some sequence (ex: some word in a text)
    Find the collection of the biggest non-overlapping elements
    
    Returns spans of elements to keep
    


    tiebreaker: elements may have a classification which is kept in the optinal
    third index of the tuple. 

    If hierarchy exists, tiebreaker is a dict which maps
    classes to priorities, ie if tie between class1 and 2, and tiebreaker
    is {class1:0, class2:1} then keep class2

    Optional annotations can be kept in rest of tuple
    
    Algorithm:
    
    list-->sort list
    keep track of last seen element (initialized to the first element)
    for each element in the list
        if the current element is outside the span of last element, it is now the last element
        if the current element is within last elements span,
        keep the longest element of the two, which is now the last element
        In case of tie, and tiebraker is specified and a third elemnt is provided,
        use that element and the tiebreaker dictionary to resolve it
    """
    
    spans = sorted(spans, key = lambda k: (k[0], k[1]))
    s,e = spans[0][0:2]
    keep = [True for _ in range(len(spans))]
    curr = 0

    for i,span in enumerate(spans[1:], start = 1):

        span = span[0:2]
        if span[0] >= s and span[0] <= e:

            length_new = span[1] - span[0]
            length_curr = e - s

            if length_new < length_curr:

                keep[i] = False

            elif length_new > length_curr:

                s,e = span
                keep[curr] = False
                curr = i

            elif tiebreaker is None: 

                raise NotImplementedError("Two entities with the same length overlap and no tiebreaker")
            
            else:

                ccurr = tiebreaker[spans[curr][2]]
                cnew = tiebreaker[spans[i][2]]

                if cnew > ccurr:

                    s,e = span
                    keep[curr] = False
                    curr = i

                else:

                    keep[i] = False
        else:

            s,e = span
            curr = i
            
    return [s for s,k in zip(spans,keep) if k]

def get_entity_tags(tags):

    """
    Get which entity tags correspond to entities
    
    returns a list of list of 2 elements: first element is token index
    of label start and second id token index for last token ofa given entity
    
    
    TODO: add more test cases
        no error thrown for entitiies that dont make sense ie
        an I after a O
        no error thrown for mixing entity types (eg B-ent1 I-ent2)
    """
    inside = False
    ret = []
    start = 0
    end = 0

    for i,t in enumerate(tags):

        if t[0] == "B":

            if inside:

                end = i - 1
                ret.append((start,end))

            start = i
            inside = True

        if t[0] == "O" and inside:

            end = i - 1
            inside = False
            ret.append((start,end))

    if inside:

        end = i
        ret.append((start,end))

    return ret

def get_span_entity(tags,span):

    """
    Given a sequence of BIO labels
    and their spans in some text
    return the spans of all entities in that text

    An entity is defined by a sequence that starts with a B and
    optionally followed by any sequence of Is
    """
    return [(span[t1][0], span[t2][1]) for t1,t2 in get_entity_tags(tags)]


def _get_span_entity(tags,span):
   
    """
    Given a sequence of BIO labels
    and their spans in some text
    return the spans of all entities in that text

    An entity is defined by a sequence that starts with a B and
    optionally followed by any sequence of Is

    TODO: add more test cases
        no error thrown for entitiies that dont make sense ie
        an I after a O
        no error thrown for mixing entity types (eg B-ent1 I-ent2)
    
    """
    inside = False
    start = span[0][0]
    end = span[0][0]
    ret = []
    
    for t,s in zip(tags,span):
        
        if t[0] == "B":
            
            if inside:
                
                ret[-1][1] = prev_end
                
            ret.append([s[0],s[1]])
            inside = True
        
        elif t[0] == "O" and inside:
            
            inside = False
            ret[-1][1] = prev_end
            
        prev_end = s[1]
        
    if inside:
        
        ret[-1][1] = s[1]
        
    return ret   

def get_span_entity_re(tags,span):

    """
    return spans of entities using regexp
    allows onlt for entity detection, not classification
    """
    tags = "".join(tags)
    res = (list(re.finditer(r"BI*", tags)))
    return [ (span[x.span()[0]][0],span[x.span()[1] - 1][1]) for x in re.finditer(r"BI*", tags)]

