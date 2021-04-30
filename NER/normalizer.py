import json

class normalizer:
    
    def __init__(self,entity_dict,preprocess = None):
        
        self.entity_dict = entity_dict
        if preprocess is None:
            
            self.preprocess = lambda x: x
            
        else:
            
            self.preprocess = preprocess
        
    def __getitem__(self,term):
        
        return self.entity_dict.get(self.preprocess(term),term)
   
def load_foodb_synonyms(direc,preprocess = lambda x: x):

    synonym_to_id = dict()

    with open(direc + "CompoundSynonym.json") as f:
        
        for i,line in enumerate(f):
            
            record = json.loads(line)
            synonym_to_id[record["synonym"]] = record["source_id"]
            
    cp_id = dict()
    with open(direc + "Compound.json") as f:
        
        for i,line in enumerate(f):
            
            record = json.loads(line)
            cp_id[record["id"]] = record["name"]
            
    synonym_to_cp = dict()
    for synonym,c_id in synonym_to_id.items():
        
        if c_id in cp_id:

            synonym_to_cp[preprocess(synonym)] = cp_id[c_id]
            
    return normalizer(synonym_to_cp, preprocess)
