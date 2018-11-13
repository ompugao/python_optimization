import numpy as np
import scipy.linalg as linalg
import logging
log = logging.getLogger(__file__)
MEPS = 1.0e-10

def lp_revised_simplex(c, A, b):
    np.seterror(divide='ignore')

    m, n = A.shape

    AI = np.hstack([A, np.identity(m)])
    c0 = np.r_[c, np.zeros(m)]

    basis = [n+i for i in range(m)] #indices
    nonbasis = [j for j in range(n)] #indices

    while True:
        y = linalg.solve(AI[:,basis].T, c0[basis])
        cc = c0[nonbasis] - np.dot(y, AI[:, nonbasis])

        if np.all(cc < MEPS): #optimization criteria
            x = np.zeros(n+m)
            x[basis] = linalg.solve(AI[:basis], b)
            log.info('Optimal Solution found')
            optval = np.dot(c0[basis], x[basis])
            return optval, x[:m]
        
        s = np.argmax(cc)
        d = linalg.solve(AI[:, basis], AI[:, nonbasis])
        if np.all(d < MEPS):
            log.warn('unbounded')
            return None, None
        bb = linalg.solve(AI[:, basis], b)
        ratio = bb/d
        ratio[ratio<-MEPS] = np.inf
        r = np.argmin(ratio)
        nonbasis[s], basis[r] = basis[r], nonbasis[s] #swap basis and nonbasis




