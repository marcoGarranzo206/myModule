import pandas as pd
import ahocorasick
import networkx as nx


def split_onto_ID(x):

    return x.split("/")[-1].split("#")[-1]

class ontology:


    """
    Class made to work with ontologies with bioportal ontology CSV files

    Functionalities include creating the Directed Acyclic Graph (DAG) induced
    by the ontology, creating an Ahocorasick automaton to search a text for terms
    from the ontology, merging ontologies (TO-DO).
    
    """
    def __init__(self, filename, *args,**kwargs):

        """
        Create ontology object from CSV bio portal ontology file

        Following attributes are created:

            DAG: directed acyclic graph induced by ontology file. Directed graph
            with edges leaving a parent and entering a child. A child inherits all information
            from the parent

            ID_NAMES: dictionary that points the ontologies ID to a set of terms
            
            ID_SYNONYMS: Same, but for synonyms
        """
        df = pd.read_csv(filename,*args,**kwargs)
        self._create_DAG(df)

    def _create_DAG(self,df):

        self.DAG = nx.DiGraph() #network created from the ontology. Nodes correspond to IDs
        self.ID_NAMES = dict()  #ID-to-terms dictionary
        self.ID_SYNONYMS = dict() #ID-to-synonyms dictionary

        for i,row in df.iterrows():

            id_, parents = split_onto_ID(row["Class ID"]), row["Parents"]

            if not pd.isna(parents):

                parents = [split_onto_ID(x) for x in parents.split("|")]

                for parent in parents:

                    self.DAG.add_edge(parent.split("/")[-1], id_)

            else:

                #if has parent, node is added via edge
                #otherwise, added here
                self.DAG.add_node(id_)


            terms = row["Preferred Label"].lower().strip().split(" / ")
            synonyms = row["Synonyms"]

            self.ID_NAMES[id_] = terms

            if not pd.isna(synonyms):

                syn = synonyms.lower().strip().split("|")
                self.ID_SYNONYMS[id_] = syn

    def get_leaves(self,synonym = True):
        


        """
        Return the leaf elements, and optionally, their synonyms, from the DAG

        Leave terms are the most descriptive terms and typically denote
        a specific entity and not an entity type or class
        """


        leaves_terms = set()
        for n,d in self.DAG.out_degree():

            if d > 0:

                continue

            try:

                leaves_terms.update(self.ID_NAMES[n])

            except KeyError:

                pass

            if synonym and n in self.ID_SYNONYMS:

                leaves_terms.update(self.ID_SYNONYMS[n])

        return leaves_terms

    def get_all_terms(self, synonym = True):

        to_ret = set()

        for n in self.DAG.nodes():

            if n in self.ID_NAMES:

                to_ret.update(self.ID_NAMES[n])

            if synonym and n in self.ID_SYNONYMS:

                to_ret.update(self.ID_SYNONYMS[n])

        return to_ret

    def make_automaton(self,only_leaves = True, synonym = True, process = lambda x:  x):

        """
        make ahocorasick automaton of terms in the ontology for fast retrieval of them
        in a text

        automaton comes from pyahocorasick implementation, check their docs for how
        to use them

        only_leaves: use only leaf nodes or all term
        synonym: use synonyms or not
        process: process the terms beforehand or not (ie lowerase)
        must be a function that takes a string and returns a string

        """
        if only_leaves:

            terms = self.get_leaves(synonym)

        else:

            terms =  self.get_all_terms(synonym)

        A = ahocorasick.Automaton()
        for idx, key in enumerate(terms):

            key = process(key)
            A.add_word(key, (idx, key))

        A.make_automaton()

        return A
