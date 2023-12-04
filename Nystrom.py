import numpy as np
import scipy.sparse as sp


def approximateEigendecomposition(L, k, n, alpha):
    """
    :param L: input Laplace-Matrix
    :param k: number of clusters
    :param n: number of samples in original dataset
    :param alpha: percentage of landmark points selected as subsample for the Nyström method from the original dataset
    
    :return H: first k Nyström approxiamted eigenvectors
    :return lam: first k eigenvalues for the Nyström approximated eigenvectors
    :return subsample_pos: indices of landmarks chosen for the subsampling in the original dataset (for experiment analysis)
    """
    
    # select initial landmark points with highest degrees
    degrees = sp.csr_matrix.diagonal(L)
    num_subsamples = int(n*alpha)
    subsample_pos = np.argpartition(degrees, -num_subsamples)[-num_subsamples:]
   
    # create matrix containing only edges chosen for the subsampling
    L_1_temp = L[:, subsample_pos]          # refers to matrix [A1, A2] in equation (3.4) in thesis 
    L_1 = L_1_temp[subsample_pos, :]        # refers to matrix A1 in equation (3.4) in thesis
    
    # compute eigenvectors of landmark points (normed to length 1)
    try:
        lam, H_1 = sp.linalg.eigsh(L_1, k, which='SM', maxiter=1000000)
    except Exception:
        print("WARNING: eigendecomposition has not converged")
        lam = np.ones(k)
        H_1 = np.ones((num_subsamples, k))
   
    # create matrix containing remaining edges
    L_2 = L_1_temp[getOppositeSample(subsample_pos, n), :]      # refers to matrix A2 in equation (3.4) in thesis 
    
    # use the Nyström extension to extrapolate eigenvectors for remaining edges
    H_2 = np.dot((L_2.dot(H_1)), np.diag(-lam**-1))  
    H_2_norm = np.linalg.norm(H_2, axis=0)
    H_2 = H_2 / H_2_norm[np.newaxis]
    
    # reassemble eigenvectors
    H = np.zeros((n, k))
    H[subsample_pos] = H_1
    H[getOppositeSample(subsample_pos, n)] = H_2
      
    return H, lam, subsample_pos


def getOppositeSample(pos, N):
    idx = np.ones(N, dtype = bool)
    idx[pos] = False
    
    opposite = []
    for i in range(0, N):
        if idx[i]:
            opposite.append(i)
    
    return opposite
    
     
def eigendecompositionNystrom(L, k, alpha):
    """
    :param L: input Laplace-Matrix
    :param k: required number of clusers
    :param alpha: fraction of points to be subsampled (between 0 and 1)
    
    :return H: k nyström approxiamted eigenvectors
    :return lam: first k eigenvalues for the Nyström approximated eigenvectors
    :return subsample_pos: array of indices that have been chosen for the subsample     
    """
    
    # calculate eigendecomposition using the Nyström method
    H, lam, subsample_pos = approximateEigendecomposition(L, k, L.shape[0], alpha)
    
    # norm approximated eigenvectors
    H_norm = np.linalg.norm(H, axis=0)
    H = np.nan_to_num(H / H_norm[np.newaxis])

    return H, lam, subsample_pos


