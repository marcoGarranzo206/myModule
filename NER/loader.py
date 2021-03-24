import en_core_web_sm
from itertools import groupby
from myModule.functions import full_listdir
from myModule.evaluate_DDI import extract_DDI_corpus
nlp = en_core_web_sm.load()
import tokenizations
import numpy as np

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

    returns a list of list of tokens and a list of list of spans

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
        
    return tokens,spans

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


def tokenize_DDICorpus(text,outputs, sentence_tokenizer, word_tokenizer):
    
    """
    outputs: list of tuples:
    
        (start index, end index, NER type)
    """
    
    docs,spans = tokenize_span(text, sentence_tokenizer,word_tokenizer)
    if not outputs:
        for sentence,span in zip(docs,spans):
              
            labels = ["O" for _ in sentence]
            yield sentence, labels,span
    
    else:
        
        curr_label_idx = 0
        curr_start, curr_end, curr_label = outputs[0]
        started = False


        for sentence,span in zip(docs,spans):


            labels = [None for _ in sentence]

            for i,(token,(token_idx, end)) in enumerate(zip(sentence,span)):

                #print(token_idx, curr_start, curr_end)
                if token_idx > curr_end:

                    started = False
                    curr_label_idx += 1
                    curr_start, curr_end, curr_label = outputs[min(curr_label_idx,len(outputs)-1)]

                if token_idx == curr_start:

                    started = True
                    labels[i] = f"B-{curr_label}"

                elif token_idx < curr_end and started:

                    labels[i] = f"I-{curr_label}"

                else: 

                    labels[i] = "O"

            yield sentence, labels, span

def load_DDI(root,text_processor,sentence_tokenizer,word_tokenizer,start_index = 0):

    """
    returns a generator of tokens,labels
    of a directory following the DDI corpus structure
    as indicated by root directory

    start_index: for DB train, ignore first file
    """

    files = sorted(full_listdir(root)[start_index:])
    grouped = groupDatasets(files, projectionDDI)
    sentences = []
    outputs = []
    spans = []

    for text,output in extract_DDI_corpus(grouped):

        #group,name,start,end = output
        output_formatted = [(x[2],x[3],x[0]) for x in output]
        text = text_processor(text)

        for tokens, labels, span in tokenize_DDICorpus(text,output_formatted, sentence_tokenizer, word_tokenizer):

            sentences.append(tokens)
            outputs.append(labels)
            spans.append(span)

    return sentences, outputs, spans

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



