from .loader import tokenize_span, put_back
import re
from nltk.tokenize import sent_tokenize
from .preprocessing import word_tokenize_sents 
import warnings

def translate_span(span,decoded, encoded):
    
    #relies that anything between words is whitespace and can be skipped
    #any partial match on a token in decoded is extended to that full token
    #TODO: use binary search? overkill for most cases, other things to do.....
    assert len(decoded) == len(encoded); "Encoded and decode spans must have same length"
    start = None
    prev_e = None
    prev_d = None
    finished = False
    end = 0
    
    for d,e in zip(decoded,encoded):
        
        if span[0] < d[1] and start is None:
            
            start = e[0]
            if d[0] < span[0]:
                
                warnings.warn(f"Extended entity started in {span} to {d[0]}")
                
        if d[1] >= span[1] and start is not None:
            
            if span[1] <= d[0]:
                
                end = prev_e[1]
                
            else:
                
                end = e[1]
                if d[1] > span[1]:
                    
                    warnings.warn(f"Extended entity ending in {span} to {d[1]}")
                
            return start,end
        
        prev_e = e
        prev_d = d
        
    else:
        
        raise ValueError("Entity not found")

def stringSearchComplicates(text,automaton, normalizer,regexp = r"(,|;|:|\(|\)|\[\])"):
   
    #TODO: allow custom sentence and word tokenizer
    transformed = re.sub(regexp, " ",text.lower())
    assert len(transformed) == len(text); "Transformed text is different length than original text, getting original spans back is impossible"
    tokens, spans, _ = tokenize_span(transformed, sent_tokenize, word_tokenize_sents)
    norm_sent = normalizer(tokens)
    
    for norm_tok, og_span in zip(norm_sent, spans):
        
        norm_sent, norm_spans = put_back(norm_tok, og_span)
        
        for res in automaton.iter(norm_sent):
    
            
            start = res[0] - len(res[1][1]) + 1
            end = res[0] + 1
            s,e = translate_span((start,end), norm_spans, og_span )
            yield (s,e, text[s:e])
