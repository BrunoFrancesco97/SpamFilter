import numpy as np

def tf_idf(features):
    ndoc = features.shape[0]
    idf = np.log10(ndoc/(features != 0).sum(0))
    return (features/100.0)*idf
