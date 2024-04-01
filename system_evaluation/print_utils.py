from tabulate import tabulate

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

        
        

