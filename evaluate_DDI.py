from collections import defaultdict
import warnings
class oneClassNER_evaluator:

    """
    Evaluate  NER system for one class of named entities

    Either something is or isnt a named entity. No granularity w.r.t to
    entity type

    Check number of terms found, not found, incorrectly identified NE
    Check which terms have been found once, which terms have been missclasied once
    """
    
    def __init__(self):
        
        self.found = set() #set of terms that were found at least once in all sentences
        self.not_found = set() #set of terms that were not picked up at least once
        self.not_in_doc = set() #set of terms deemed an NER but were not
        self.doc_terms = set() #set of named entities from those evaluated

        self.n_found = 0
        self.n_not_found = 0
        self.n_not_in_doc = 0

    def eval_sentence(self,sentence,terms,found,verbose = False):
        
        """
        
        Terms, found: list of tuple
        Tuple triples
        
        First element: terms
        Second element: start pos
        Third element: length
        """
        
        terms_sentence = {x[0] for x in terms}
        terms_found = {x[0] for x in found}
        
        idxs_sentence = {(x[1], x[2]) for x in terms}
        idxs_found = {(x[1], x[2]) for x in found}
        
        self.doc_terms.update(terms_sentence)
        self.found.update(terms_sentence.intersection(terms_found))
        self.not_found.update(terms_sentence - terms_found )
        self.not_in_doc.update(terms_found - terms_sentence)

        if verbose:

            print(sentence)
            print(terms)
            print(found)
            print()
        
        self.n_found += len(idxs_sentence.intersection(idxs_found))
        self.n_not_found += len(idxs_sentence - idxs_found)
        self.n_not_in_doc += len(idxs_found - idxs_sentence) 

    def never_found(self):
        
        return self.not_found - self.found

    def FP_terms(self):

        """
        incorrect classifcation 
        Might be due to word polysemy if using 
        a dictionary.
        
        Given a set of entity names
        Find which entity names were incorrectly classified
        as that entity
        """

        return self.not_in_doc.intersection(self.doc_terms)

def extract_BRAT_corpus(files):
    

    for ann_file, text_file in files:

        with open(text_file, "r+") as f:

            text = f.readlines()
            text = "".join(text)

        with open(ann_file, "r+") as f:

            annotations = []

            for line in f:

                annotations.append(line[:-1].split("\t"))

        entities = []
        interactions = []

        for i,ann in enumerate(annotations):

                if ann[0][0] != "T":

                    R, Type_args = ann[0:2]
                    if len(ann) > 2:

                        warnings.warn(f"Field with {len(ann)} columns found in annotation file. Skipping past column 2")
                    Type_args = Type_args.split(" ")
                    Type, arg1, arg2 = Type_args
                    arg1 = arg1.split(":")[1]
                    arg2 = arg2.split(":")[1]
                    interactions.append((R,Type, arg1, arg2))
                    continue


                T, G_span, term = ann
                splitted = G_span.split(";")

                group, start,end = splitted[0].split(" ")
                start,end = int(start), int(end)
                entities.append((T,group,term,start,end))

                for span in splitted[1:]:

                    span = span.split(" ")
                    start,end = int(start), int(end)
                    entities.append((T,group,term,start,end))
   
        yield text, entities, interactions


def eval_DDI_corpus(files,evaluator, NER_extractor, processor = lambda x: x, *args, **kwargs):

    for text, ground_truth in extract_DDI_corpus(files):

        processed_ground_truth = [(processor(x[1]), x[2], x[3]) for x in ground_truth]
        extracted = NER_extractor.extract(text,*args,**kwargs)
        extracted = [(x[0],x[1], x[1] + x[2]) for x in extracted]
        evaluator.eval_sentence(text, processed_ground_truth, extracted)


def eval_FDA_drug_labels(sentences, evaluator,automaton):

    """
    TO FINISH
    
    Evaluate NER and RE from FDA drug labels as annotated by TAC and NLM 
    for 2018 challenge (https://bionlp.nlm.nih.gov/tac2018druginteractions/ ,
    https://lhce-brat.nlm.nih.gov/NLMDDICorpus.htm)
    
    
    Sentence: iterator of xml "Sentence" XML tags
    Contains the sentence text which will be processed by the evaluator
    Contains tags for different entities and relations
    """
    
    for sentence in sentences:


        res = list(automaton.iter(sentence.text.lower().replace(",", " ")))
        res = [ ( x[1][1].strip(), x[0] - len(x[1][1]) + 1, len(x[1][1]) - 2) for x in res  ]

        to_eval = []
        for tag in sentence.find_all("Mention"):

            if tag["type"] == "Precipitant":

                vals = tag["span"].split(";")
                terms = tag["str"].split(" | ")

                for term,val in zip(terms, vals):

                    start, end = val.split(" ")           
                    start,end = int(start), int(end)
                    to_eval.append((term, start, end))

        #print(to_eval, res)
        evaluator.eval_sentence(sentence, to_eval, res)

def eval_found(term_types,terms_found,process = lambda x: x):

    """
    Evaluate an ontologys ability to find terms in a document/corpus. Checks
    which terms have been found
    
    returns: dictionary of keys
    
    term_types: dictionary of keys: type of term (drug, sympton etc..) and 
    values of set of terms
    
    terms_found:  terms_found using various ontologies. dict of keys:ontology name,
    values, set of terms found
    
    process: if any processing has to be done on term_types terms to be compatible
    with
    
    """
    
    ret_found = defaultdict(dict)
    ret_not_found = defaultdict(dict)
    
    for type_,terms in term_types.items():
        
        terms = {process(term) for term in terms}
        
        for ontology,found in terms_found.items():
            
            ret_found[ontology][type_] = found.intersection(terms)
            ret_not_found[ontology][type_] = terms - found
    
    return ret_not_found, ret_found
