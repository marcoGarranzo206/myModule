import warnings
import en_core_web_sm
from collections import defaultdict
from itertools import groupby
from myModule.functions import full_listdir
from myModule.evaluate_DDI import extract_BRAT_corpus
nlp = en_core_web_sm.load()
import tokenizations
import numpy as np


def put_back(token_list,span):

    """
    given a list of tokens and their spans in the text
    reconstruct text so adjacent tokens are as separated by as many 
    whitespaces as indicated by spans, and spans are the new
    absolute spans
    """

    if len(token_list) == 1:

        return token_list[0], span

    offset = [end[0] -start[1] for (start,end) in zip(span[:-1], span[1:])] + [0]
    new_spans = [None for _ in range(len(span))]
    mod_tokens = [None for _ in range(len(token_list)*2)]
    start = 0

    for i,(token,off) in enumerate(zip(token_list,offset)):

        end = start + len(token)
        mod_tokens[2*i] = token
        mod_tokens[2*i + 1] = " "*off
        new_spans[i] = (start,end)
        start = end + off

    return "".join(mod_tokens), new_spans
    
def safe_tokenize(token_list):


    #tokenizations doesnt word well when a " is replaced by `` or ''
    #such as in treebank tokenizer 
    safe_dict = {'``': '"',  "''": '"'}

    return [safe_dict[t] if t in safe_dict else t for t in token_list]

def tokenize_span(text,sentence_tokenizer,word_tokenizer):
    """
    tokenize a text and return the tokens as well as their span (start,end) in the
    original text

    tokenizes the text into sentences and then each sentence into tokens

    returns a list of list of tokens and a list of list of spans of those tokens and the sentences

    word_tokenizer: tokenizes list of sentences. returns list of list of tokens
    """
    sentences = sentence_tokenizer(text) 

    #separation between sentences may be variable, take into accout when adding to the span
    so_helper = tokenizations.get_original_spans(sentences,text)
    sentence_offset = np.array([so_helper[i+1][0] - so_helper[i][1] for i in range(len(so_helper) - 1) ] + [0])
    #print(sentence_offset)
    
    tokens = word_tokenizer(sentences)
    safe_tokenized = [safe_tokenize(token_list) for token_list in tokens]
    lengths = np.array(list(map(len, sentences))) + sentence_offset
    lengths = np.cumsum(lengths)
    spans = [tokenizations.get_original_spans(t,s) for t,s in zip(safe_tokenized,sentences)]
    #check if all tokens have been aligned (not None)
    #if not, and have two conseutive unaligned tokens, abort
    #else, infer spans from tokens

    for i in range(len(spans)):

        if spans[i][0] is None:

            if spans[i][1] is None:

                print(text[ spans[i][0][0]:spans[i][-1][1] ])
                print(spans[i])
                print(tokens[i])
                raise NotImplementedError("Cannot perform alingment with two consecutive unaligned tokens")

            spans[i][0] = (0, spans[i][1][0] - 1)

        if spans[i][-1] is None:

            if spans[i][-2] is None:

                print(text[ spans[i][0][0]:spans[i][-1][1] ])
                print(spans[i])
                print(tokens[i])
                raise NotImplementedError("Cannot perform alingment with two consecutive unaligned tokens")

            spans[i][-1] = (spans[i][-2][1] + 1, len(sentences[i]))

        try:

            spans[i][1:-1] = [(spans[i][j][0], spans[i][j][1]) if spans[i][j] is not None else (spans[i][j-1][1]+1, spans[i][j+1][0]-1 ) for j in range(1,len(spans[i])-1) ]

        except TypeError:

            print(text[ spans[i][0][0]:spans[i][-1][1] ])
            print(spans[i])
            print(tokens[i])
            raise NotImplementedError("Cannot perform alingment with two consecutive unaligned tokens")

    #each sentence is tokenized separately, with the span being
    #calculated separatly for each sentence
    #for each token in each sentence, add to the span 
    #the length of all previous sentences
    #print(tokens)
    #print(spans)
    for i in range(1, len(spans)):
        
        for w in range(len(spans[i])):
            
            spans[i][w] = (spans[i][w][0] + lengths[i-1], spans[i][w][1] + lengths[i-1]) 
        
    return tokens,spans,so_helper

def projectionDDI(x):

    return "".join(x.split(".")[:-1])

def groupDatasets(x, projection):

    """
    Function to group files so that 
    input (text) files and output (annotations)
    are together

    returns a list of tuples. First element of tuple
    is input, the rest are output files
    """

    x = sorted(x, key = projection)
    x = [sorted(list(it)) for k, it in groupby(x, projection)]
    
    return x


def tokenize_BRATCorpus(text,outputs, sentence_tokenizer, word_tokenizer):
    
    """
    TODO: labelling strict vs loose?
    entities: list of tuples:
    
        (id,start index, end index, NER type)

    interactions: list of tuples:

        (interaction_id, type, id1,id2)

    returns generator of tuples:

        (tokenized sentences, labels, spans, tid_loc ) 
        TODO: explain outputs
        IMP: assumes output labels are sorted based on beginning position and non-overlapping!
    """
    
    docs,spans,_ = tokenize_span(text, sentence_tokenizer,word_tokenizer)
    if not outputs:
        for sentence,span in zip(docs,spans):
              
            labels = ["O" for _ in sentence]
            yield sentence, labels,span, dict()
    
    else:
        
        curr_label_idx = 0
        tid, curr_label, name, curr_start, curr_end = outputs[0]
        started = False


        for sentence,span in zip(docs,spans):

            tid_loc = dict()
            labels = [None for _ in sentence]

            for i,(token,(token_idx, end)) in enumerate(zip(sentence,span)):

                #print(token_idx, curr_start, curr_end)
                if token_idx > curr_end:

                    started = False
                    curr_label_idx += 1
                    tid, curr_label, name, curr_start, curr_end = outputs[min(curr_label_idx,len(outputs)-1)]

                if ((token_idx >= curr_start) and (token_idx <= curr_end) or (curr_start < end <= curr_end))\
                        and not (started):

                    started = True
                    tid_loc[tid] = i
                    labels[i] = f"B-{curr_label}:{tid}"

                elif token_idx <= curr_end and started:

                    labels[i] = f"I-{curr_label}:{tid}"

                else: 

                    labels[i] = "O"

            yield sentence, labels, span, tid_loc

def groupFiles(root, start_index = 0):

    files = [f for f in full_listdir(root) if (f.endswith(".ann") or f.endswith(".txt"))]
    files = files[start_index:]
    return groupDatasets(files, projectionDDI)

def load_BRAT(root,text_processor,sentence_tokenizer,word_tokenizer,start_index = 0):

    """
    returns a generator of tokens, token labels, entity interactions 
    per document
    of a directory following the DDI corpus structure
    as indicated by root directory

    start_index: for DB train, ignore first file
    """

    files = groupFiles(root, start_index)
    doc_sentences = []
    total_labels = []
    interaction_ids = [] #defaultdict(list) #key: sentence idx. values: list of pair of ids
    #j = 0 # sentence index
    inters = [] # for debugging. will be removed if problem is solved
    tot_spans = []

    for doc_id, (text,entities,interactions) in enumerate(extract_BRAT_corpus(files)):
        
        if not all(entities[i][3] <= entities[i+1][3] for i in range(len(entities)-1)): #check if sorted

            entities.sort(key = lambda k: k[3])

        text = text_processor(text)
        curr_interaction = 0
        labels_found = 0
        curr_sents = []
        curr_labels = []
        
        interaction_ids.append(defaultdict(list))
        doc_sentences.append([])
        total_labels.append([])
        tot_spans.append([])
        inters.append(interactions)
        
        if interactions:
      
            #in case interactions arent sorted by appearance
            #map entities to index pos
            #sort interactions based on entitie pos of first entity
            ent_id = {e[0]:e[3] for e in entities}
            interactions.sort(key = lambda k: ent_id[k[2]])
            _,type_,arg1,arg2 = interactions[curr_interaction]
        

        else:
            
            type_,arg1,arg2 = "", "", ""


        j = 0
        for sents,labels,spans,t_ids in tokenize_BRATCorpus(text, entities,sentence_tokenizer,\
                                        word_tokenizer):

            labels_found +=  sum([1 if label[0] == "B" else 0 for label in labels ])#for debugging
            doc_sentences[-1].append(sents)
            total_labels[-1].append(labels)
            tot_spans[-1].append(spans)
            #curr_sents.append( [ (t,l,s)for (t,l,s) in zip(sents, labels,spans)] ) #for debugging

            while (arg1 in t_ids) and (arg2 in t_ids) and not (curr_interaction == len(interactions)):

                curr_interaction = curr_interaction + 1
                interaction_ids[-1][j].append((type_,t_ids[arg1],t_ids[arg2]))

                if (curr_interaction == len(interactions)):

                    break

                _,type_,arg1,arg2 = interactions[curr_interaction]
            
            j += 1
        
        if labels_found < len(entities):
            
            #TODO: try to solve the missing entities/missing interactions problem
            #has to do with incorrectly split tokens
            #some entities are included as part of a (non entity word-entity word) token so
            #the beginning span does not line with any token span. This should be fixed 
            #with
            #if (token_idx >= curr_start) and (token_idx <= curr_end) and not (started):
            #other tokens include two entities ie entity1-entity2, so that token gets tagged
            #as the first entity. In one case, there is an interaction between those 2 entities
            #which cannot get picked up
            warnings.warn(f"{files[doc_id][0]} has {len(entities)} but found only {labels_found}")


    return doc_sentences,total_labels,interaction_ids, tot_spans #,inters

def write_formatted(sentences,labels,file_,extra_cols = None, start_sent = 0, sep = "\t", **kwargs):

    """
    write sentences in a typical format for NER
    Each row has minimum 3 columns: sentence id, token and label
    Extra columns can indicate additional info ie POS, lemma...
    tokens in a sentence are supposed to be ordered:
        the first token of the sentence is its first token, and so on


    sentences,labels: list of length N
    
    ith index has length of sentence i
    jth element of sentences[i] is the jth token
    jth element of labeks[i] is the jth tokens output

    extra_cols: either None or list of lenth N + 1
   
    0th index: names of columns, length C. 
    ith index has length of sentence i
    jth element of extra_cols[i]:

        container of length C
        contains the attribute values of token j in sentence i

    """

    print("SENTENCE_INDEX","TOKEN","LABEL",sep = sep, end = "", file = file_, **kwargs)

    if extra_cols is not None:

        names = extra_cols[0]
        print(sep + sep.join(names), sep = "", end = "", file = file_, **kwargs)

    print(file = file_, **kwrags)

    for i, (sentence,label) in enumerate(zip(sentences, labels), start = start_sent):

        for word,label in zip(sentence,label):

            print(i,word,label,sep = sep, end = "", file = file_, **kwargs)

            if extra_cols is not None:

                print(sep + sep.join(map(str, extra_cols[i])), sep = "", end = "", file = file_, **kwargs )

            print(file = file_, **kwargs)



