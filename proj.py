import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    '''
    d = original dimension
    k = projected dimension
    '''
    return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
    _, d= X.shape
    return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
    '''
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    '''
    n, d = X.shape
    assert(d>=k)
    _, _, Vh = np.linalg.svd(X)
    V = Vh.T
    return V[:, :k]

def pca_proj(X, k):
    '''
	compute projection of matrix X
	along its first k principal components
	'''
    # P = P.dot(P.T)
    P = my_pca(X, k)
    return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    '''
	Fitting a k dimensional feature set obtained
	from random projection of X, versus y
	for binary classification for y in {-1, 1}
	'''

    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)

    # predict y
    y_pred=line.predict(X_test.dot(J))

    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
    '''
	Fitting a k dimensional feature set obtained
	from PCA projection of X, versus y
	for binary classification for y in {-1, 1}
	'''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)

    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)

     # predict y
    y_pred=line.predict(X_test.dot(P))


    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)


######## LOADING THE DATASETS #########

# to load the data:
files = ['data1.npz','data2.npz','data3.npz']
n_trials = 10  # to average for accuracies over random projections

# Using PCA and Random Projection for:
# Visualizing the datasets
for i in files:
    data = np.load(i)
    X = data['X']
    y = data['y']
    n, d = X.shape

    mrand = random_proj(X, 2)
    mpca = pca_proj(X, 2)

    plt.figure(1)
    plt.plot(mpca[:,0], mpca[:,1], c = 'orangered', markersize = 1, alpha=0.5,label = "PCA")
    plt.plot(mrand[:,0], mrand[:,1], c = 'aquamarine', markersize = 1, alpha=0.5,label = "random")
    legend = plt.legend(loc=1, shadow=True)
    plt.show()

    plt.figure(2)

    plt.scatter(mpca[:,0],mpca[:,1], c = 'orangered', s = 1, alpha=0.5, label = "PCA")
    plt.scatter(mrand[:,0], mrand[:,1], c = 'aquamarine', s = 1, alpha=0.5, label = "random")
    legend = plt.legend(loc=1, shadow=True)
    plt.show()
