import numpy as np
from random import choice
import requests
from collections import defaultdict

class pubChemRet:

    """
    to search if compound_name is in pubChem:

        domain = "compound", namespace = "name" in __init__
        search: compound_name, operation = "cids"

    to search for a classification of a CID:

        domain = compound, namespace = CID, output = JSON in __init__
        search: CID, operation = "classification"
    
    self._content contains the results from these operations
    more info at https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest¶
    """

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, domain, namespace,output = "JSON"):
        
        self.domain = domain
        self.namespace = namespace
        self.output = output
        
    def modDomain(self, newDomain):
        
        self.domain = newDomain
        
    def modNamespace(self, newNamespace):
        
        self.namespace = newNamespace
        
    def modOutput(self, newOutput):
        
        self.output = newOutput
        
    def search(self, ID, params = {}, operation = None):
        
        self.ID = ID
        if operation is None:
            
            url = pubChemRet.base_url + "/" + self.domain + "/"+ self.namespace + "/" + self.ID + "/"  + self.output
            
        else:
            
            url =  pubChemRet.base_url + "/" + self.domain + "/"+ self.namespace + "/" + self.ID + "/" + operation  + "/" + self.output
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        self.__dict__.update(response.__dict__)

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

class default_bidict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(default_bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(default_bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(default_bidict, self).__delitem__(key)

def random_selection(universe):
    
    return np.random.choice(universe)


class discreteGeneticSolver:
    
    def __init__(self, p,crossover_method, universe,length, fitness, \
                 parent_contribution = 0.5,pop = 100, selection = random_selection, universe_type = "equal" ):
        
        """
        Coarse implementation of genetic algorithm for discrete decision process
        
        There is an objective function to evaluate (fitness) with decision values that can take
        on a finite discrete set of values
        
        An "individual" here is just one assingment of the decision variables
        His "DNA" is composed of the decision variables.
        
        A population of size pop starts randomly guessing the answers. Fitness function evaluates them
        Based on this fitness function, each individual has a probability to mate
        
        During mating, an offspring is produced with part of the "DNA" from one parent and another part 
        from another parent
        
        p: probability of mutation
        crossover_method: how "offspring" are produced from parents
        for the moment, two choices:
            midpoint: select a point. All decisions variables with index up until that point is inherited
            from one parent, and the others from the other parent
            crossover: each decision has a probability to be inherited from one parent or the other
            
        parent_contribution: mating happens randomly between p1 andp2. It is the % of DNA
        you want from parent 1.
        
        length: how long the "DNA strand" is, ie the number of decision variables
        
        Universe and universe_type: the values each decision variable tcan take. If equal, each decision variable
        can take the same values. If not equal, universe[i] is the values variable i can take, so len(universe)
        must equal length
        
        """
        
        self.p = p
        self.crossover_method = crossover_method
        self.universe = universe
        self.fitness = fitness
        self.parent_contribution = parent_contribution
        self.selection = selection
        
        if universe_type not in ("equal", "specific"):
            
            raise ValueError(f"universe_type must be equal or specific, not {universe_type}")
            
        if universe_type == "specific":
            
            if len(universe) != length:
                
                raise ValueError(f"len(universe) is {len(universe)}, not length!")
            
            self.pop = [[None]*len(universe) for _ in range(pop)]
            for i in range(pop):

                for j in range(len(universe)):

                    self.pop[i][j] = np.random.choice(universe[j])

        else:
            
            self.pop = [np.random.choice(universe, length) for _ in range(pop)]
            
        self.universe_type = universe_type
        
    def solve(self, n_iters, verbose = True):
        
        for j in range(n_iters):
            
            fitness = self.eval_fitness()
            max_i = np.argmax(fitness)

            if j == 0:

                max_fitness = fitness[max_i]
                max_response = self.pop[max_i]
                
            elif fitness[max_i] > max_fitness:

                max_fitness = fitness[max_i]
                max_response = self.pop[max_i]
            
            if verbose:

                print(fitness[max_i])
            #print("".join(self.pop[max_i]), fitness[max_i])
            #print("".join(string_to_guess))
            self.pop = [self.select(fitness) for _ in range(len(self.pop))]
            
            for i in range(len(self.pop)):
                
                self.mutate(i)
        fitness = self.eval_fitness()
        max_i = np.argmax(fitness)
        
        if fitness[max_i] > max_fitness:

            max_fitness = fitness[max_i]
            max_response = self.pop[max_i]
        return max_response
        
    def eval_fitness(self):
        
        return [self.fitness(x) for x in self.pop]
    
    def select(self, fitness):
        
        prob = fitness/np.sum(fitness)
        p1,p2 = np.random.choice(a = len(self.pop), p = prob), np.random.choice(a = len(self.pop), p = prob)
        
        return self.crossover(self.pop[p1],self.pop[p2])
    
    def crossover(self,p1,p2):
        
        l = len(p1)
        child = [0 for _ in range(l)]
        
        if hasattr(self.parent_contribution, "__call__"):
 
            c = self.parent_contribution()
        else:
            
            c = int(np.ceil(l*self.parent_contribution))
            
        if self.crossover_method == "midpoint":
            
            child[:c] = p1[:c]
            child[c:] = p2[c:]
           
        elif self.crossover_method == "crossover":
            
            helper_parent = [p1,p2]
            for i in range(l):
                
                child[i] = helper_parent[np.random.choice(a = 2, p = [c, 1 - c])][i]
        #print(child)        
        return child
        
        
    def mutate(self,individual):
        
        for i in range(len(self.pop[individual])):
            
            if np.random.random() < self.p:
                
                if self.universe_type == "equal":
                    
                    self.pop[individual][i] = self.selection(self.universe)
                else:
                    
                    self.pop[individual][i] = self.selection(self.universe[i])


