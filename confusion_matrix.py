import numpy as np
import pickle as pkl
from pdb import set_trace as st

class CM:
    def __init__(self, cm_file, obs, labels):
        '''
        labels: an input dictionary such as {1: "ped", 2:"obs", ...}
        '''
        self.cm_file = cm_file
        self.obs = obs
        self.labels = labels
        self.cm, self.param_cm = self.read_confusion_matrix()
        self.assert_labels()
        self.C = self.construct_confusion_matrix_dict(self.cm) # Class-based confusion matrix
        self.dist_param_C = dict() # Proposition based, distance parametrized confusion matrix
        self.n = len(self.labels)
        self.eps = 1e-3

    def assert_labels(self):
        '''
        Function to check whether the correct confusion matrix is being read
        '''
        assert len(list(self.labels.keys())) == len(self.cm[0])

    def get_canonical(self):
        n = len(self.conf_matrix[0][0])
        self.canonical_cm = np.zeros((n,n))
        for k, v in self.conf_matrix.items():
            self.canonical_cm += v # Total class based conf matrix w/o distance

    def read_confusion_matrix(self):
        self.conf_matrix = pkl.load(open(self.cm_file, "rb" ))
        
    
    def parametrized_confusion_matrix(self):
        for k, cm in self.param_cm.items():
            self.dist_param_C[k] = self.construct_confusion_matrix_dict(cm)
        
    def construct_confusion_matrix_dict(cm):
        '''
        Constructing the confusion matrix from stored data
        '''
        C = dict()

        for ktrue, label_true in self.labels.items():
            total_data_true_label = np.sum(cm[:,ktrue-1])

            if total_data_true_label!=0.0:
                for kpred, label_pred in self.labels.items():
                    C[label_pred, label_true] = cm[kpred-1, ktrue-1]/total_data_true_label
            else:
                for kpred, label_pred in self.labels.items():
                    C[label_pred, label_true] = 0.0
        return C