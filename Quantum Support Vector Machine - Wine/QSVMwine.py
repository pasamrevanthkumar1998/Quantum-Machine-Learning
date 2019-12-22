import numpy as np
from qiskit import IBMQ
IBMQ.load_account()
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def wine(training_size, test_size, n, plot_data=True):
    """ returns wine dataset """
    class_labels = [r'A', r'B']

    data, target = datasets.load_wine(True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=test_size, random_state=7)

    
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    
    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}

    if plot_data:
        if not HAS_MATPLOTLIB:
            raise NameError('Matplotlib not installed. Plase install it before plotting')
        for k in range(0, 2):
            label = 'Class-B' if k==1 else 'Class-A'
            plt.scatter(sample_train[label_train == k, 0][:training_size],
                        sample_train[label_train == k, 1][:training_size],label=label)

        plt.title("PCA dim. reduced Wine dataset")
        plt.legend()
        plt.show()
        

    return sample_train, training_input, test_input, class_labels

from qiskit.aqua.utils import split_dataset_to_data_and_labels

n = 2 
training_dataset_size = 20
testing_dataset_size = 10

sample_Total, training_input, test_input, class_labels = wine(training_dataset_size, testing_dataset_size, n)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
print(class_to_label)

from qiskit.aqua.input import ClassificationInput
from qiskit import Aer
from qiskit.aqua import run_algorithm

params = {
    'problem': {'name': 'classification', 'random_seed': 10598},
    'algorithm': { 'name': 'QSVM' },
    'backend': {'name': 'qasm_simulator', 'shots': 1024},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
}

backend = Aer.get_backend(name='qasm_simulator')

algo_input = ClassificationInput(training_input, test_input, datapoints[0])
result = run_algorithm(params, algo_input, backend=backend)

print("ground truth:    {}".format(datapoints[1]))
print("prediction:      {}".format(result['predicted_labels']))
print("predicted class: {}".format(result['predicted_classes']))
print("accuracy:        {}".format(result['testing_accuracy']))