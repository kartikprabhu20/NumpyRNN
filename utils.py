"""

    Created on 13/01/21 12:54 AM 
    @author: Kartik Prabhu

"""
import numpy as np

def init_weights(type,range1,range2,dim1,dim2):
    if type=='uniform':
        return np.random.uniform(range1,range2,([dim1,dim2]))
    if type=='normal':
        #  range1=mu; range2=sigma
        return np.random.normal(range1, range2, size=([dim1,dim2]))

"""
Simple one-hot encoding
"""
def embedding(X,seq_len, vocab_size):

    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        x[ind] = np.eye(vocab_size)[batch]
    return x