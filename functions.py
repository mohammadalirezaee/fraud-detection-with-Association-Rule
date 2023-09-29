# Description: This file contain the functions used in the main file.

def get_kandid(dataset):
    """This function generates combinations of column names from a dataset,
        creating pairs of columns within groups of 20, and avoiding repetitions."""
    lis = list(dataset.columns)
    a = []
    merged_list = []

    for i in range(29):
        a.append(lis[20*i : 20*(i+1)])

    kandid_list = []
    for i in range (29):
        for j in range (29):
            if i < j:
                kandid_list.append(list(product(a[i], a[j])))
    for lst in kandid_list:
        merged_list.extend(lst)

    return merged_list


def k_candidate_list(k,dataset_columns_list):
    """This function generates combinations of column names from a dataset,"""
    k_candidate = [comb for comb in combinations(dataset_columns_list, k)]
    return k_candidate


def add_class_to_tuple(tup): 
    """This function adds the class column to the tuples of the candidate list.
        It returns a list of tuples with the class column added.
        The class column is added to the end of the tuple."""
    if isinstance(tup,set):
        list_filtered_sup = list(tup)
    else:       
        list_filtered_sup = list(tup.keys())
    candidate_fraud = []
    for i in list_filtered_sup:
        i = list(i)
        i.append('class_fraud')
        i = tuple(i)
        candidate_fraud.append(i)
    return candidate_fraud

def confidence_for_K_candidate_list(K_candidate_list, dataset, filtered_sup =True):
    """This function calculates the confidence for each tuple in the candidate list.
        It also calculates the support for each tuple in the candidate list.
        It returns two dictionaries, one with the confidence and the other with the support.
        The confidence is calculated as the number of times the tuple appears in the dataset"""
    confidence_dic = {}
    support_dic = {}
    for pairs in K_candidate_list:
        objects = np.array(pairs[:-1]).reshape(1,-1)[0]
        clas =  pairs[-1]
        class_object_support = len(dataset[dataset[objects].sum(axis=1) + dataset[clas] == len(objects) + 1])
        
        if filtered_sup == True:
            object_support = filtered_sup[tuple(objects)]
        else:
            object_support = len(dataset[dataset[objects].sum(axis=1) == len(objects)])

        if object_support == 0:
            confidence = 0
            class_object_support = 0
        else:
            confidence = class_object_support / object_support

        confidence_dic[pairs] =  confidence
        support_dic[pairs] = class_object_support
    return confidence_dic , support_dic

def add_class_to_tuple(tup): 
    """This function adds the class column to the tuples of the candidate list.
        It returns a list of tuples with the class column added.
        The class column is added to the end of the tuple."""
    if isinstance(tup,set):
        list_filtered_sup = list(tup)
    else:       
        list_filtered_sup = list(tup.keys())
    candidate_fraud = []
    for i in list_filtered_sup:
        i = list(i)
        i.append('class_normal')
        i = tuple(i)
        candidate_fraud.append(i)
    return candidate_fraud

def k_1_common(lis):  

    three_canidate = []
    for  index1 , first  in enumerate(lis):
        for index2 , second in enumerate(lis):
            if  index1 < index2 :
                lis1 = list([first,second])
                merged_list = []
                for lst in lis1:
                    merged_list.extend(lst)
                if len(set(merged_list)) == len(first) + 1:
                    three_canidate.append(tuple(set(merged_list)))
    return set(three_canidate) 

def remove_class(lis): 
    """This function removes the class column from the tuples of the candidate list."""
    final_list = []   
    for i in lis:
        if 'class_normal' in i:
            id = i.index('class_normal')
            i = list(i)
            i.pop(id)
            i = tuple(i)
            final_list.append(i)
        else:
            id = i.index('class_fraud')
            i = list(i)
            i.pop(id)
            i = tuple(i)
            final_list.append(i)
    return final_list

def labeling(dataset):
    """This function labels the dataset with the rules found by the algorithm.
        It returns a list with the labels.
        The labels are 1 if the rule is true and 0 if the rule is false."""
    label_list = []
    for index , row in dataset.iterrows():
        V12_1 = row['V12_1']
        V17_1 = row['V17_1']
        V11_20 = row['V11_20']
        V9_1 = row['V9_1']
        V3_1 = row['V3_1']
        V7_1 = row['V7_1']

        if (row['V12_1'] == 1 and row[ 'V17_1'] == 1 or
            row['V7_1'] == 1 and row[ 'V11_20'] == 1  and row['V17_1'] == 1 or
            row['V7_1'] == 1 and row[ 'V3_1'] == 1  and row['V17_1'] == 1 or
            row['V9_1'] == 1 and row[ 'V17_1'] == 1 and row[ 'V11_20'] == 1 or
            row['V12_1'] == 1 and row[ 'V17_1'] == 1 and row['V7_1'] == 1 or
            row['V9_1'] == 1 and row[ 'V7_1'] == 1 and row[ 'V17_1'] == 1):
            label = 1
            label_list.append(label)
        else: 
            label = 0
            label_list.append(label)
    return label_list

def calsiffication(predicted_label_list , first_dataset):
    """This function calculates the accuracy, precision, recall and f1_score of the algorithm.
        It returns the accuracy, precision, recall and f1_score.
        It also returns the true positive, false positive, true negative and false negative.
        The true positive is the number of frauds that were correctly classified as frauds.
        The false positive is the number of normal transactions that were incorrectly classified as frauds.
        The true negative is the number of normal transactions that were correctly classified as normal.
        The false negative is the number of frauds that were incorrectly classified as normal.
        The accuracy is the number of correct predictions divided by the total number of predictions.
        The precision is the number of correct positive predictions divided by the total number of positive predictions.
        The recall is the number of correct positive predictions divided by the total number of positive values.
        The f1_score is the harmonic mean of the precision and recall."""
    grand_truth = list(first_dataset['Class'])
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for predict , real in zip(predicted_label_list,grand_truth):
        if predict == 1 and real == 1:
            TP += 1
        elif predict == 1 and real == 0:
            FP += 1
        elif predict == 0 and real == 0:
            TN += 1
        elif predict == 0 and real == 1:
            FN += 1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy , precision , recall , f1_score , TP , FP , TN , FN

