from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def ista_solve_hot( A, d, la_array ):
    # ista_solve_hot: Iterative soft-thresholding for multiple values of
    # lambda with hot start for each case - the converged value for the previous
    # value of lambda is used as an initial condition for the current lambda.
    # this function solves the minimization problem
    # Minimize |Ax-d|_2^2 + lambda*|x|_1 (Lasso regression)
    # using iterative soft-thresholding.
    max_iter = 10**4
    tol = 10**(-3)
    tau = 1/np.linalg.norm(A,2)**2
    n = A.shape[1]
    w = np.zeros((n,1))
    num_lam = len(la_array)
    X = np.zeros((n, num_lam))
    for i, each_lambda in enumerate(la_array):
        for j in range(max_iter):
            z = w - tau*(A.T@(A@w-d))
            w_old = w
            w = np.sign(z) * np.clip(np.abs(z)-tau*each_lambda/2, 0, np.inf)
            X[:, i:i+1] = w
            if np.linalg.norm(w - w_old) < tol:
                break
    return X


def num_errors(A):
    errors = 0
    for i in range(A):
        for j in range(A[0]):
            if A[i][j] > 0.000001:
                errorserrors + 1
    return errors



print("Program is starting...")

cancerFiles = loadmat("BreastCancer.mat")
lambdas = [0.000001, 0.000002, 0.00001, 0.00002, 0.0001, 0.0002, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1, 2, 10, 20]
solution = ista_solve_hot(cancerFiles['X'][0:100], cancerFiles['y'][0:100], lambdas)
#goal is to plot 2-norm of Aw* − d squared vs w*

ASol = cancerFiles['X'][0:100]
wSol = solution
lamSol = lambdamed 
dSol = cancerFiles['y'][0:100]


w_star = ista_solve_hot(ASol, dSol, lambdas)
subtraction =  np.matmul(ASol, w_star) - dSol



#plt.plot(np.linalg.norm(solution[0:295], ord=1), np.linalg.norm(np.matmul(cancerFiles['X'], solution) - cancerFiles['y'], ord=2))
plt.xlabel('||w*||(1)')
plt.ylabel('||Aw∗ − d||(2)^2');
plt.plot(w_star[0:100], subtraction)


print("Program is done")


lambda1 = [0.000001]
lambda2 = [0.000002]
lambda3 = [0.00001]
lambda4 = [0.00002]
lambda5 = [0.0001]
lambda6 = [0.0002]
lambda7 = [0.001]
lambda8 = [0.002]
lambda9 = [0.01]
lambda10 = [0.02]
lambda11 = [0.1]
lambda12 = [0.2]
lambda13 = [1]
lambda14 = [2]
lambda15 = [10]
lambda16 = [20]
