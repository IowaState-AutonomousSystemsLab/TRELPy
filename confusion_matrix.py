import numpy as np
import pickle as pkl
from pdb import set_trace as st
from collections import OrderedDict as od

class ConfusionMatrix:
    def __init__(self, cm_generator, obs, labels, label_type="class"):
        '''
        Attributes:
        cm_generator: GenerateConfusionMatrix object that was used to compute the confusion matrix. 
        This object carries all the information on 
        obs: List of observations
        labels: an input dictionary such as {1: "ped", 2:"obs", ...}
        n: number of labels
        cm_file: Default is None. String containing the location of where the (distance-parametried) canonical confusion matrix was saved
        prop_cm_file: Default is None. String containing the location of where the (distance-parametried) proposition-labeled confusion matrix was saved
        '''
        self.obs = obs
        self.cm_generator = cm_generator
        self.labels = labels
        self.n = len(self.labels)
        self.prop_cm_file = None
        self.cm_file = None
        self.confusion_matrix = None # Canonical confusion matrix stored in file
        self.prop_confusion_matrix = None # Proposition confusion matrix stored in file
        self.model_info = None # ToDo: this needs to be set.

    def assert_labels(self, label_type):
        '''
        Function to check whether the correct confusion matrix is being read
        '''
        if label_type == "class":
            assert len(list(self.labels.keys())) == len(self.confusion_matrix[0])
        
        if label_type == "prop":
            assert len(list(self.labels.keys())) == len(self.prop_confusion_matrix[0])
        

    def get_canonical(self):
        self.canonical_cm = np.zeros((self.n,self.n))
        for k, v in self.confusion_matrix.items():
            self.canonical_cm += v # Total class based conf matrix w/o distance

    def read_confusion_matrix(self, file, label_type="class"):
        with open(file, "rb") as f:
            if label_type == "class":
                if self.cm_file is None:
                    self.cm_file = file
                self.confusion_matrix = pkl.load(f)
            else:
                if self.prop_cm_file is None:
                    self.prop_cm_file = file
                self.prop_confusion_matrix = pkl.load(f)
        f.close()
    
    def read_from_file(self, cm_file, label_type):
        self.read_confusion_matrix(cm_file, label_type)
        self.assert_labels(label_type)

    def set_confusion_matrix(self, cm, label_type="class"):
        if label_type == "class":
            self.confusion_matrix = cm
        else:
            self.prop_confusion_matrix = cm
    
    def save_confusion_matrix(self, file, label_type="class"):
        with open(file, "wb") as f:
            if label_type == "class":
                if self.cm_file is None:
                    self.cm_file = file
                pkl.dump(self.confusion_matrix, f)
            else:
                if self.prop_cm_file is None:
                    self.prop_cm_file = file
                pkl.dump(self.prop_confusion_matrix, f)
        f.close()

    def construct_confusion_matrix_dict(cm):
        '''
        Constructing the confusion matrix from stored data to normalize
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