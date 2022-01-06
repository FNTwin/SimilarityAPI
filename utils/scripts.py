import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def kernel(x1,x2,n):
    return ((x1@x2) / (( x1@x1 ) * (x2@x2))**0.5)**n

def distancekernel(x1,x2,n=1):
    try:
        return ( 2.0 - 2.0 * kernel(x1,x2,n)) ** 0.5
    except:
        return 0

def reduce_dimension(data):
    n=10
    ax_n=1
    pca_data=np.loadtxt("data\SOAP_full.txt")
    pca = PCA(n_components=n).fit(pca_data)
    return normalize(pca.transform(data), axis= ax_n) 


def compose_new_simil(reduced_data=None):
    if reduced_data is None:
        return np.loadtxt("data\Similarity_matrix.txt")
    computed_simil=np.loadtxt("data\SOAP_reduced.txt") #Maybe wrong file
    return np.vstack((computed_simil,reduced_data))

def build_ordered_sim_matrix(data):
    r=[]
    for i in data:
        c=[]
        for j in data:
            c.append(distancekernel(i,j, 1))
        r.append(c)

    similarity_matrix=np.asarray(r)
    b = np.sum(similarity_matrix, axis = 1)
    idx = np.argsort(b, axis=0)[::-1]

    ordered_similarity_matrix=similarity_matrix[idx,:]
    ordered_similarity_matrix=ordered_similarity_matrix[:,idx]
    return ordered_similarity_matrix, idx

def build_plot(similarity_matrix, idx):

    ticks=np.array(["NP-4^o","NP-4^b","NP-2^b","NP-2^o","NP-5^ ","NP-3^ ","NP-1^ ", "NP-2^340K","NP-4^340K",
                    "NP-5/6^ ","NP-4/6^g","NP-4/6^o","NP-2/6^b","NP-2/6^o","NP-4/6^","NP-1/6^ ",
                    "NP-4/6^340K", "NP-2/6^340K", "SUBMITTED"])
    if len(idx) != 19:
        ticks = ticks[:-1]
    sorted_ticks=ticks[idx]

    return similarity_matrix, sorted_ticks
    
