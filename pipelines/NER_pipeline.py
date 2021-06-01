from copy import deepcopy
from myModule.data_augmentation.data_augmentation_NER import data_augmentator as dA
from myModule.NER.evaluate import eval_results
from myModule.NER.loader import load_BRAT
from myModule.NER.feature_extraction import extractor
import numpy as np
from NERDA.models  import NERDA
from myModule.NER import sequenceModels

def unravel_list_of_list(x):

    return [ z for y in x for z in y]

class NER_loader:


    def __init__(self, word_processor, sent_tokenize, word_tokenize, word_normalizer, dap):

        self.word_processor = word_processor
        self.sent_tokenize = sent_tokenize
        self.word_tokenize = word_tokenize
        self.word_normalizer = word_normalizer
        self.dp = dap

    def _load_corpus(self, train_folder, test_folder, validation_split, detect = False):

        sents,ents,_,_= load_BRAT(train_folder, self.word_processor, self.sent_tokenize, self.word_tokenize, self.word_normalizer)
        test_sents,test_ents,_,_= load_BRAT(test_folder, self.word_processor, self.sent_tokenize, self.word_tokenize, self.word_normalizer)
        
        sents = unravel_list_of_list(sents)
        ents = unravel_list_of_list(ents)
        ents = [[e.split(":")[0] for e in sent] for sent in ents]

        test_sents = unravel_list_of_list(test_sents)
        test_ents = unravel_list_of_list(test_ents)
        test_ents = [[e.split(":")[0] for e in sent] for sent in test_ents]

        if detect:

            ents = get_first_letter(ents)
            test_ents = get_first_letter(test_ents)
        
        shuffling = np.random.choice([0,1], p = [1-validation_split, validation_split], size = len(sents))
        train_sents = [s for s,t in zip(sents, shuffling) if t == 0]
        train_ents = [e for e,t in zip(ents, shuffling) if t == 0]

        valid_sents = [s for s,t in zip(sents, shuffling) if t == 1]
        valid_ents = [e for e,t in zip(ents, shuffling) if t == 1]
       
        augmentator = dA(train_sents, train_ents)

        total_train_sents = deepcopy(train_sents)
        total_train_ents = deepcopy(train_ents)

        augmented_texts, augmented_ents = augmentator.augment_MR(self.dp["MR"]["p"],self.dp["MR"]["n"])
        total_train_sents.extend(augmented_texts)
        total_train_ents.extend(augmented_ents)

        augmented_texts, augmented_ents = augmentator.augment_LwTR(self.dp["LwTR"]["p"],self.dp["LwTR"]["n"])
        total_train_sents.extend(augmented_texts)
        total_train_ents.extend(augmented_ents)


        augmented_texts, augmented_ents = augmentator.augment_SR(self.dp["SR"]["p"],self.dp["SR"]["n"],
                self.dp["SR"]["hypernyms_SR"], self.dp["SR"]["hyponyms_SR"])
        total_train_sents.extend(augmented_texts)
        total_train_ents.extend(augmented_ents)
        
        return total_train_sents, total_train_ents, valid_sents, valid_ents, test_sents, test_ents

    def __call__(self, string):

        for sents,_,spans,_ in tokenize_BRATCorpus(string, [], self.sent_tokenizer,\
                                        self.word_tokenize):

            yield sents,spans

class NER_charbilstm:
    #TODO: ALL
    def __init__(self, loader_pipeline,PARAMS):

        self.loader_pipeline = loader_pipeline
        self.cbilstm_params = {}
        self.training_hyperparameters = {}
        self.dropout = dropout
        self.extraction = {}

    def create_model(self,train_folder, test_folder, validation_split, detect = False):

        train_sents, train_ents, valid_sents, valid_ents, test_sents, test_ents = \
        self.loader_pipeline._load_corpus(train_folder, test_folder, validation_split, detect)
        extractor = extractor(self.extraction["extractors"], self.extraction["kwargs"])
        self.model = sequenceModels.cbilstm_extra_features(extractor, **self.cbilstm_params)
        self.model.fit(train_sents, valid_sents, train_ents, valid_ents, **self.training_hyperparameters)

class NER_NERDA:

    def __init__(self,loader_pipeline, PARAMS):

        self.loader_pipeline = loader_pipeline
        self.training_hyperparameters = {'epochs' : int(PARAMS["epochs"]),
                                    'warmup_steps' : int(PARAMS["warmup_steps"]),
                                    'train_batch_size': 256,
                                    'learning_rate': PARAMS["learning_rate"]}
        self.transformer = PARAMS["transformer"]
        self.dropout = PARAMS["dropout"]

    def create_model(self,training, validation, tag_scheme):


        self.model = NERDA(
        dataset_training = training,
        dataset_validation = validation,
        tag_scheme = tag_scheme,
        tag_outside = 'O',
        transformer = self.transformer,
        dropout = self.dropout,
        hyperparameters = self.training_hyperparameters
        )
    
    def train(self,train_folder,test_folder,validation_split, detect = False):

        train_sents, train_ents, valid_sents, valid_ents, test_sents, test_ents = \
        self.loader_pipeline._load_corpus(train_folder, test_folder, validation_split, detect)
        
        tag_scheme = list(set(e for sent in train_ents for e in sent if e != "O"))
        training = {"sentences": train_sents, "tags": train_ents}
        validation = {"sentences": valid_sents, "tags": valid_ents}
        test = {"sentences": test_sents, "tags": test_ents}
        self.create_model(training,validation,tag_scheme)
        self.model.train()
        valid_res = self.model.predict(valid_sents)

        return eval_results(valid_ents, valid_res)

    def __call__(self, string):

        tokens = []
        spans = []

        for t,s in self.loader_pipeline(string):
            tokens.append(t)
            spans.append(s)

        return self.model.predict(tokens)

    def eval(self,tokens,labels):

        pred = self.model.predict(tokens)
        return eval_results(labels,pred)
