from myModule.objects import bidict
import abc

class mapper(abc.ABC):
    
    """

    Base class. Separate subclass for tokens/chars due to how
    preprocessing is different
    
    """

    def __init__(self, transformer):
    
        """
        Map tokens/characters to indexes after some user defined
        transformation. Useful for embeddings.
        
        Two extra tokens indixes are implicit: a padd and unk 
       
        if an element which wasnt in the training data is found,
        it is assigned the unk token index

        padd token: last index + 1
        unk token: last index + 2
        """
        
        self.mapping = bidict()
        self.transformer = transformer
                    
    def __len__(self):
        
        return len(self.mapping)
    
    def train(self,text):
        
        """
        given some text, extract all unique elements 
        and map them to an index
        """
        
        uniq = set(text)
        uniq = {self.transformer(u) for u in uniq}
               
        for i,u in enumerate(uniq,start = 1):
            
            self.mapping[u] = i
        self.pad = i
        self.unk = i + 1
    
    @abc.abstractmethod
    def __getitem__(self,sequence):

        pass

class char_mapper(mapper):
    
    def __getitem__(self, sequence):
        
        sequence = self.transformer(sequence)
        return [ self.mapping.get(s,self.unk) for s in sequence ]
        

            
class token_mapper(mapper):
        
    def __getitem__(self, sequence):
        
        sequence = [self.transformer(t) for t in sequence]
        return [ self.mapping.get(s,self.unk) for s in sequence ]
