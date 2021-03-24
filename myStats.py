import numpy as np
from scipy import stats
from collections import Counter
from scipy.stats import rankdata

def fdr(p_vals):

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

def binom(query, ref_counts, N, adj = "BH", classification = None):
    
    """
    query: iterable of nodes
    classification: mapping of node name to classification
    ref counts: counts of each class in reference
    N: total number of nodes
    """
    
    counts = Counter()
    n = 0
    pvals = dict()
    
    for node in query:

        if classification is not None:
            
            if node in classification:

                counts.update(classification[node])
        
        else:
            
            counts.update([node])

        n += 1
        
    ret = {}
    classes = ref_counts.keys()
    pvals = np.zeros(len(classes))
    signs = np.zeros(len(classes))
    for i, class_ in enumerate(classes):
        
        p = ref_counts[class_]/N
        E = p*n
        pvals[i] = stats.binom_test(counts[class_], n = n, p = p)
        signs[i] = np.sign(counts[class_]-E) 
        
    if adj == "BH":
        
        pvals = fdr(pvals)

    return {class_ : (p,s) for class_,p,s in zip(classes, pvals, signs)}

