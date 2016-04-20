import numpy as np 
from scipy import linalg
from sklearn import linear_model

N = 200
x = np.random.normal(0, 4, N)
y1 = 3 * x + np.random.normal(3, 1, N)
y2 = 3 * x + np.random.normal(-3, 1, N)
y_avg = np.mean(np.append(y1, y2))

# features x and y are rows in this np.matrix

X = np.append([x, y1 - y_avg], [x, y2 - y_avg], 1)
X_t = X.T

# y1 group 0; y2 group 1

truth = np.append(np.ones([1, N]), np.zeros([1, N]), 1)


# Now try dimension reduciton with SVD

(W, s, V_t) = linalg.svd(X)

# both transformed representations are the same before trimming:

S = linalg.diagsvd(s, len(X), len(V_t))
print X_t.shape, u.shape, S.shape, v_t.shape
#np.max(abs(X.T * np.matrix(W) - np.matrix(V_t).T * np.matrix(S).T))

# Now work with the transformed coordinates.  It might not have been clear
# from above what the transformed coordinate system was. We can get there
# by either

X_prime = np.matrix(V_t).T * np.matrix(S).T
x_prime = np.asarray(X_prime.T[0])
y_prime = np.asarray(X_prime.T[1])

# Linearly classifiable in 1-d? Try all new basis directions (extremes of variation)
# Min variation - Training along y-dim nearly perfect
lr = linear_model.LogisticRegression()
ypt = np.asarray(y_prime.T)
print ypt
lr.fit(ypt, truth[0])
print 'y_prime', lr.score(ypt, truth[0])
lr.predict(ypt)

xpt = np.asarray(x_prime.T)
lr.fit(xpt, truth[0])
print 'x_prime', lr.score(ypt, truth[0])
lr.predict(ypt)