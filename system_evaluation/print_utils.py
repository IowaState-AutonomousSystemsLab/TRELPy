from tabulate import tabulate
from pdb import set_trace as st

def save_cm_latex(cm, cm_dict, fn):
    table_C = print_cm_for_latex(cm, cm_dict)
    with open(fn, "w") as f:
        f.write(table_C)

def save_param_cm_latex(cm, cm_dict, fn):
    print_param_cm_for_latex(cm, cm_dict, fn)

def print_param_cm_for_latex(cm, cm_dict, file):
    # TO-DO: make it generic
    text_file = open(file, "w")
    for distance_bin, bin_cm in cm.items():
        text_file.write("\n")
        text_file.write("Printing confusion matrix from distance range " +str(distance_bin))
        text_file.write("\n")
        cm_k = print_cm_for_latex(bin_cm, cm_dict)
        text_file.write(cm_k)
        text_file.write("\n")
    text_file.close()

def print_cm_for_latex(cm, cm_dict):
    headers=['']
    predictions = []
    matrix = dict()
    for key, label in cm_dict.items():
        if label not in headers:
            headers.append(label)
        try:
            if label not in matrix.keys():
                matrix[label] = cm[key]
        except:
            # For proposition labeled
            if tuple(label) not in matrix.keys():
                matrix[tuple(label)] = cm[key]
    
    for label, pred_row in matrix.items():
        predictions.append([label, *pred_row])

    table_C = tabulate(predictions, headers=headers, tablefmt='latex')
    return table_C


def print_param_cm(cm):
    # TO-DO: make it generic
    for distance_bin, bin_cm in cm.items():
        print("Printing confusion matrix from distance range ", distance_bin)
        print_cm(bin_cm)

def print_cm(cm):
    print(" ")
    headers=['']
    predictions = []
    matrix = dict()
    for key, prob in cm.items():
        pred, true = key
        if true not in headers:
            headers.append(true)
        if pred not in matrix.keys():
            matrix[pred] = dict()
        matrix[pred][true] = prob
    
    for pred_class, pred_mat in matrix.items():
        pred_row = []
        for true, pred in pred_mat.items():
            pred_row.append(pred)
        predictions.append([pred_class, *pred_row])

    table_C = tabulate(predictions, headers=headers, tablefmt='latex')
    print(table_C)
    
    


        
        

