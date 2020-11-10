import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
import csv

edges_file = open('wisconsin_edges.csv', "r")
nodes_file = open('wisconsin_nodes.csv', "r")

# create a dictionary where nodes_dict[i] = name of wikipedia page
nodes_dict = {}
for line in nodes_file:
    nodes_dict[int(line.split(',',1)[0].strip())] = line.split(',',1)[1].strip()

node_count = len(nodes_dict)


# create adjacency matrix
A = np.zeros((node_count, node_count))
for line in edges_file:
    from_node = int(line.split(',')[0].strip())
    to_node = int(line.split(',')[1].strip())
    A[to_node, from_node] = 1.0
    
    

    
#add 0.001 to each entry 
print("addition started")
rows = 5482
columns = 5482

for i in range(rows):
    for j in range(columns):
        A[i, j] = A[i,j] + 0.001
print("addition finished")


#normalization
print("normalization started")
A = A/A.sum(axis=0)
print("normalization finished")

#eigendecomposition
Anp = np.array(A)
print("eigendecomposition started")

lam, V = eigs(csc_matrix(A), k = 1)
print("lam")
print(lam)
print("V")
print(V)

#VλV^-1
Vinv = inv(V)
Lambda = np.diag(lam)


    
eigDecomp = np.matmul(V, Lambda)

eigDecomp = np.matmul(eigDecomp, Vinv)

print("eigendecomposition finished")

        
    

## Add code below to (1) prevent traps and (2) find the most important pages     
# Hint -- instead of computing the entire eigen-decom
# you can compute just the first eigenvector with:
# s, E = eigs(csc_matrix(A), k = 1)