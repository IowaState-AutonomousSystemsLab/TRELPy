import pickle as pkl
import numpy as np

# Read confusion matrix from nuscnes file:
def read_confusion_matrix(cm_fn, prop_dict_file=None):
    conf_matrix = pkl.load( open(cm_fn, "rb" ))
    for k,v in conf_matrix.items():
        n = len(conf_matrix[k][0])
        break
    cm = np.zeros((n,n))
    for k, v in conf_matrix.items():
        cm += v # Total class based conf matrix w/o distance
    
    if prop_dict_file:
        prop_dict = pkl.load(open(prop_dict_file, "rb" ))
        return cm, conf_matrix, prop_dict
    else:
        return cm, conf_matrix


# Script for confusion matrix of pedestrian
# Make this cleaner; more versatile
# C is a dicitionary: C(["ped", "nped"]) = N(observation|= "ped" | true_obs |= "nped") (cardinality of observations given as pedestrians while the true state is not a pedestrian)
# Confusion matrix for second confusion matrix
def confusion_matrix(conf_matrix_file):
    C = dict()
    param_C = dict()
    cm, param_cm = read_confusion_matrix(conf_matrix_file)
    print(cm)
    C = construct_confusion_matrix_dict(cm)
    for k, cm in param_cm.items():
        param_C[k] = construct_confusion_matrix_dict(cm)
    return C, param_C # Parametrized cm

def construct_confusion_matrix_dict(cm):
    C = dict()
    total_ped = np.sum(cm[:,0])
    total_obs = np.sum(cm[:,1])
    total_ped_obs = np.sum(cm[:,2])
    total_emp = np.sum(cm[:,3])
    if total_ped!=0.0:
        C["ped", "ped"] = cm[0,0]/total_ped
        C["obs", "ped"] = cm[1,0]/total_ped
        C["ped,obs", "ped"] = cm[2,0]/total_ped
        C["empty", "ped"] = cm[3,0]/total_ped
    else:
        C["ped", "ped"] = 0.0
        C["obs", "ped"] = 0.0
        C["ped,obs", "ped"] = 0.0
        C["empty", "ped"] = 0.0

    if total_obs!=0.0:
        C["ped", "obs"] = cm[0,1]/total_obs
        C["obs", "obs"] = cm[1,1]/total_obs
        C["ped,obs", "obs"] = cm[2,1]/total_obs
        C["empty", "obs"] = cm[3,1]/total_obs
    else:
        C["ped", "obs"] = 0.0
        C["obs", "obs"] = 0.0
        C["ped,obs", "obs"] =0.0
        C["empty", "obs"] = 0.0

    if total_ped_obs!=0.0:
        C["ped", "ped,obs"] = cm[0,2]/total_ped_obs
        C["obs", "ped,obs"] = cm[1,2]/total_ped_obs
        C["ped,obs", "ped,obs"] = cm[2,2]/total_ped_obs
        C["empty", "ped,obs"] = cm[3,2]/total_ped_obs
    else:
        C["ped", "ped,obs"] = 0.0
        C["obs", "ped,obs"] = 0.0
        C["ped,obs", "ped,obs"] = 0.0
        C["empty", "ped,obs"] = 0.0

    if total_emp!=0.0:
        C["ped", "empty"] = cm[0,3]/total_emp
        C["obs", "empty"] = cm[1,3]/total_emp
        C["ped,obs", "empty"] = cm[2,3]/total_emp
        C["empty", "empty"] = cm[3,3]/total_emp
    else:
        C["ped", "empty"] = 0.0
        C["obs", "empty"] = 0.0
        C["ped,obs", "empty"] =0.0
        C["empty", "empty"] = 0.0

    return C

def new_confusion_matrix(cm_file, prop_dict_file):
    C = dict()
    param_C = dict()
    cm, param_cm, prop_dict = read_confusion_matrix(cm_file, prop_dict_file)
    print(cm)
    C = new_construct_confusion_matrix_dict(cm, prop_dict)
    for k, cm in param_cm.items():
        param_C[k] = new_construct_confusion_matrix_dict(cm, prop_dict)
    return C, param_C # Parametrized cm

def new_construct_confusion_matrix_dict(cm, prop_dict):
    C = dict()
    for k, v in prop_dict.items():
        if "empty" in v:
            jempty = k
        elif "ped" in v and "obs" not in v:
            jped = k
        elif "ped" not in v and "obs" in v:
            jobs = k
        elif "ped" in v and "obs" in v:
            jpedobs = k
        else:
            print("Error in parsing propositions")
            st()
    total_ped = np.sum(cm[:,jped])
    total_obs = np.sum(cm[:,jobs])
    total_ped_obs = np.sum(cm[:,jpedobs])
    total_emp = np.sum(cm[:,jempty])
    if total_ped!=0.0:
        C["ped", "ped"] = cm[jped, jped]/total_ped
        C["obs", "ped"] = cm[jobs, jped]/total_ped
        C["ped,obs", "ped"] = cm[jpedobs, jped]/total_ped
        C["empty", "ped"] = cm[jempty,jped]/total_ped
    else:
        C["ped", "ped"] = 0.0
        C["obs", "ped"] = 0.0
        C["ped,obs", "ped"] = 0.0
        C["empty", "ped"] = 0.0

    if total_obs!=0.0:
        C["ped", "obs"] = cm[jped,jobs]/total_obs
        C["obs", "obs"] = cm[jobs,jobs]/total_obs
        C["ped,obs", "obs"] = cm[jpedobs,jobs]/total_obs
        C["empty", "obs"] = cm[jempty,jobs]/total_obs
    else:
        C["ped", "obs"] = 0.0
        C["obs", "obs"] = 0.0
        C["ped,obs", "obs"] =0.0
        C["empty", "obs"] = 0.0

    if total_ped_obs!=0.0:
        C["ped", "ped,obs"] = cm[jped,jpedobs]/total_ped_obs
        C["obs", "ped,obs"] = cm[jobs,jpedobs]/total_ped_obs
        C["ped,obs", "ped,obs"] = cm[jpedobs,jpedobs]/total_ped_obs
        C["empty", "ped,obs"] = cm[jempty,jpedobs]/total_ped_obs
    else:
        C["ped", "ped,obs"] = 0.0
        C["obs", "ped,obs"] = 0.0
        C["ped,obs", "ped,obs"] = 0.0
        C["empty", "ped,obs"] = 0.0

    if total_emp!=0.0:
        C["ped", "empty"] = cm[jped,jempty]/total_emp
        C["obs", "empty"] = cm[jobs,jempty]/total_emp
        C["ped,obs", "empty"] = cm[jpedobs,jempty]/total_emp
        C["empty", "empty"] = cm[jempty,jempty]/total_emp
    else:
        C["ped", "empty"] = 0.0
        C["obs", "empty"] = 0.0
        C["ped,obs", "empty"] =0.0
        C["empty", "empty"] = 0.0

    return C

# Script for confusion matrix of pedestrian
# Varying the precision/recall confusion matrix values
def confusion_matrix_ped2(prec, recall):
    C = dict()
    tp = recall*100
    fn = tp/prec - tp
    tn = 200 - fn
    C["ped", "ped"] = (recall*100)/100.0
    C["ped", "obs"] = (fn/2.0)/100.0
    C["ped", "empty"] = (fn/2.0)/100.0

    C["obs", "ped"] = ((1-recall)*100.0/2)/100.0
    C["obs", "obs"] = (tn/2*4.0/5)/100.0
    C["obs", "empty"] = (tn/2*1/5)/100.0

    C["empty", "ped"] = ((1-recall)*100/2)/100.0
    C["empty", "obs"] = (tn/2*1.0/5)/100.0
    C["empty", "empty"] = (tn/2*4.0/5.0)/100.0
    tol = 1e-4
    assert(abs(C["ped", "ped"] + C["obs", "ped"] + C["empty", "ped"] - 1.0) < tol)
    assert(abs(C["ped", "obs"] + C["obs", "obs"] + C["empty", "obs"]- 1.0)< tol)
    assert(abs(C["ped", "empty"] + C["obs", "empty"] + C["empty", "empty"]- 1.0) < tol)

    return C
