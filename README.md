# myModule
A collection of scripts written over the years. As of now will probably not be understood by anyone else.
Will try to organize it into more modules and probably delete it afterwards.

functions: python functions I use quite often and decided to dump here.
objects: python objects I use quite often and decided to dump here. Includes a bidirectional dictionary and a genetic algorithm solver.
bktreeAhupp: bk tree code from https://github.com/ahupp/bktree/ for efficient spell checking of a large dictionary
evaluate_DDI: code to evaluate a DDI 2013 challenge. As of now only evaluates NER detection.
extraPrograms: to tokenize and lemmatize scientific texts using chemTok and bioLemmatizer, written in java, when using python
graphs: some functions to work with netwrokx graphs
myStats: FDR and a function to do binomial testing
ontology: objects to work with ontologies downlaoded in csv files from NCBIO portal. Includes the ability to scan texts for terms in the ontology and to normalize entities via their synonyms
NER: module for NER. Includes scripts for loading DDI extraction data, a bidirectional cnn crf for NER and a flexible module for processing raw text. Also includes a embedder using BERT models.
RE: modulefor relationship extraction. Extract relationships between entities using lstm trees. Text must be parsed using a syntax tree according to XXX
