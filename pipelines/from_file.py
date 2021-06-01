import sys
from nltk.tokenize import sent_tokenize
from myModule.NER.preprocessing import chemtok, word_tokenize_sents, bio_lemmatize, stem,WordNetLemmatizerFromNLTK,snowball
from .NER_pipeline import NER_NERDA, NER_loader
import re

def do_nothing(x):

    return x
def remove_special(text):

    return re.sub(r'[^A-Za-z0-9\.]+', ' ', text)

mapper = {
        "None": do_nothing,
        "remove_special": remove_special,
        "chemtok_special":chemtok({"clm": True}),
        "chemtok_normal": chemtok({}),
        "wordnet_tokenize": word_tokenize_sents,
        "wordnet_lemmatize": WordNetLemmatizerFromNLTK(),
        "bioLemma": bio_lemmatize,
        "porterStemmer": stem,
        "snowballStemmer": snowball("english")
        }
def create_pipeline_from_dict(PARAMS):

    #-loader creator-#
    wp = PARAMS["word_processor"]
    wt = PARAMS["word_tokenize"]
    wn = PARAMS["word_normalizer"]
    
    n_LwTR= int(PARAMS["n_LwTR"])
    p_LwTR= PARAMS["p_LwTR"]

    n_MR= int(PARAMS["n_MR"])
    p_MR= PARAMS["p_MR"]

    n_SR= int(PARAMS["n_SR"])
    p_SR= PARAMS["p_SR"]
    hypernyms_SR= PARAMS["hypernyms_SR"]
    hyponyms_SR = PARAMS["hyponyms_SR"]
   
    da = {"LwTR": {"n": n_LwTR, "p": p_LwTR},
            "MR": {"n": n_MR, "p": p_MR},
            "SR": {"n": n_SR, "p": p_SR, "hypernyms_SR": hypernyms_SR,
                "hyponyms_SR": hyponyms_SR}}

    loader = NER_loader(mapper[wp], sent_tokenize, mapper[wt], mapper[wn],
            da)

    model = PARAMS["model"]
    if model == "transformer":

        transformer_hyperparameters = \
         {"epochs": PARAMS["epochs"],\
          'warmup_steps' : int(PARAMS["warmup_steps"]),\
          'train_batch_size': 256,\
          'learning_rate': PARAMS["learning_rate"],\
          "transformer": PARAMS["transformer"],\
          "dropout" : PARAMS["dropout"]}
        return NER_NERDA(loader, transformer_hyperparameters)

if __name__ == "__main__":

    import json
    params_file = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]
    with open(params_file, "r") as f:

        PARAMS = json.load(f)

    pipeline = create_pipeline_from_dict(PARAMS)
    valid = pipeline.train(train,test,PARAMS["validation_split"])
    for k,v in valid.items():

        print(k, v["strict"]["f1"])
