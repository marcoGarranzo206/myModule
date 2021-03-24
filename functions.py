import inspect
import re
import shutil
import os
import tarfile
from time import time
from functools import wraps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def unroll_str_col(df, col, internal_sep): 

                
    return [term for element in df[col] for term in str(element).split(internal_sep)]

def plot_str_col(df, col, internal_sep, limit = 35):

    count = Counter(unroll_str_col(df,col,internal_sep))
                
    vals = np.array(list(count.values()))
    order = np.argsort(vals)
    names = list(count.keys())
    #print(len(count))
    
    if len(count) > limit:
        
        other = np.sum(vals[order[::-1]][limit:])
        keys = list(count.keys())
        keys = [keys[i] for i in order[::-1]]
        to_remove = keys[limit:]
        for k in to_remove:
            
            count.pop(k, None)
            
        count["other"] = other
        #This can possibly be optimized
        vals = np.array(list(count.values()))
        order = np.argsort(vals)
        names = list(count.keys())
    
    ptypes = list(count.keys())
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize = (10,20))
    plt.barh(y = [names[x] for x in order], width = vals[order])
    plt.show()
    
    return count

def jaccard(set1,set2, denom = "union"):

    num = len(set1.intersection(set2))
    if denom == "union":

        denom = len(set1.union(set2))

    elif denom == "min":

        denom = min(len(set1),len(set2))
    
    elif denom == "max":

        denom = max(len(set1),len(set2))

    else:

        raise ValueError(f"denom must be union, min or max, not {denom}")

    return num/denom

def full_listdir(directory):

    if not directory.endswith("/"):

        directory += "/"

    return [directory + x for x in os.listdir(directory)]


def timer(func):
    
    import time
    
    @wraps(func)
    def wrapper(*args,**kwargs):
        
        t1 = time.time()
        result = func(*args,**kwargs)
        t2 = time.time()
        print(f"{func.__name__} ran in {t2-t1} seconds")
        
        return result
    
    return wrapper


def tar_untar(direc,regexp,func,**kwargs):
    
    """
    pass a function on a bunch of tared files whose names 
    follow some regex
    
    untar files in a helper directory
    do the function on untared files/directories, save the return in a variable
    remove the untared files
    remove helper directory
    return the variable
    """
    files = [direc + x for x in  os.listdir(direc)]
    files = re.findall(regexp,"\n".join(files))
    ret = {}
    
    #something needs to be done about this
    #the point is to untar in an empty directory
    try:
       
        name = ".taredFiles_"+str(time())
        os.makedirs(name)
        
    except FileExistsError:
        
        pass
    
    #extract which arguments from kwargs belong to tarfile.open and which
    #to user function
    tar_args = [k for k, v in inspect.signature(tarfile.open).parameters.items()]
    tar_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in tar_args}
    

    func_args = [k for k, v in inspect.signature(func).parameters.items()]
    func_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in func_args}
    
    try:
        
        for f in files:

            print(f)
            #print(os.listdir(name))            
            tarf = tarfile.open(f,**tar_dict)
            tarf.extractall(path = name)
            tarf.close()


            filename =  os.listdir(name)[0]
            filename =  name + "/" + filename
            ret[f] = func(filename,**func_dict)
            shutil.rmtree(filename)
            
    finally:  
        
        shutil.rmtree(name)
        
    return ret
