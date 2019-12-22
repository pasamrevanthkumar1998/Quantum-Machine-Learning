import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit import IBMQ
IBMQ.load_account()


df = pd.read_csv("kmeans_data.csv",
    usecols=['Feature 1', 'Feature 2', 'Class'])

df.head()

df["Class"] = pd.Categorical(df["Class"])
df["Class"] = df["Class"].cat.codes

data = df.values[:, 0:2]
category = df.values[:, 2]

k = 3

n = data.shape[0]

c = data.shape[1]


mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean


centers = np.array([[-0.25, 0.2], [0, -0.1], [0.25, 0.35]])
print(centers)


colors=['green', 'blue', 'black']
for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)

def point_centroid_distances(point, centroids):
    
    
    phi_list = [((x + 1) * pi / 2) for x in [point[0], centroids[0][0], centroids[1][0], centroids[2][0]]]
    theta_list = [((x + 1) * pi / 2) for x in [point[1], centroids[0][1], centroids[1][1], centroids[2][1]]]

    
    qreg = QuantumRegister(3, 'qreg')

    
    creg = ClassicalRegister(1, 'creg')

    qc = QuantumCircuit(qreg, creg, name='qc')

    
    backend = Aer.get_backend('qasm_simulator')

    
    results_list = []

    
    for i in range(1, 4):
        
        qc.h(qreg[2])

        
        qc.u3(theta_list[0], phi_list[0], 0, qreg[0])           
        qc.u3(theta_list[i], phi_list[i], 0, qreg[1]) 

        
        qc.cswap(qreg[2], qreg[0], qreg[1])
        
        qc.h(qreg[2])

        
        qc.measure(qreg[2], creg[0])

        
        qc.reset(qreg)

        
        job = execute(qc, backend=backend, shots=5000)
        result = job.result().get_counts(qc)
        try:
            results_list.append(result['1'])
        except:
            results_list.append(0)


    return results_list

centers_old = np.zeros(centers.shape) 
centers_new = deepcopy(centers) 

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)
upper_error = error + 1


while (error + 0.02) < upper_error:
    
    
    distances = np.array(list(map(lambda x: point_centroid_distances(x, centers), data)))

    
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    upper_error = deepcopy(error)
    error = np.linalg.norm(centers_new - centers_old)
    if error < 0.02:
        break
centers_new


for i in range(n):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)

