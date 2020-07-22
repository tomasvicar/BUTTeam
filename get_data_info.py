import os
from itertools import islice
import numpy as np
import scipy.io as io
import csv
import pandas
import glob
import json






def enumerate_labels(path_data, table_scored):
    """Fucntion for generation of label representation at whole dataset provided for training.
    Table includes true/false values for all 27 diagnoses."""
    file_list = glob.glob(path_data + r"/**/*.hea", recursive=True)
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'],
                          ['427172004', '17338001']]
    lbs_to_change = [equivalent_classes[0][1], equivalent_classes[1][1], equivalent_classes[2][1]]

    list_of_labels = list(table_scored.keys())
    labels_presence = {}
    # for file_idx, file_n in enumerate(islice(file_list, 1000)):
    for file_idx, file_n in enumerate(file_list):
        with open(file_n, "r") as file:
            for line_idx, line in enumerate(file):
                if line_idx == 15:
                    diagnosis = line.replace('#Dx: ', '').replace('\n', '')
                    codes = diagnosis.split(",")
                    for code_idx, code in enumerate(codes):
                        if code == lbs_to_change[0]:
                            codes[code_idx] = equivalent_classes[0][0]
                        if code == lbs_to_change[1]:
                            codes[code_idx] = equivalent_classes[1][0]
                        if code == lbs_to_change[2]:
                            codes[code_idx] = equivalent_classes[2][0]
                    codes_int = [int(i) for i in codes]
                    word_pom = []
                    for word in list_of_labels:
                        word_pom.append(word in codes_int)
                    # labels_presence.append(word_pom)
                    labels_pom = file_n.split('/')
                    labels_presence[labels_pom[-1]] = word_pom
                    line_idx = 0
                    break

        file.close()

    with open('true_false_labels_scored_alldata.json', 'w') as fp:
        json.dump(labels_presence, fp)

    return labels_presence


def sub_dataset_labels_sum(list_of_paths):
    
    with open('true_false_labels_scored_alldata.json') as fp:
        table_all = json.load(fp)
    
    #"""Counts the sum of labels in selected dataset for the weights computation.
    #Inputs: list of header files paths (e.q. 'C:\Users\ronzhina\CinC-2020\data\PhysioNetChallenge2020_Training_2.tar\Training_2\Q0001.hea', str),
    #total table of label representation (all database, dict)."""

    list_of_file_names = []
    for path in list_of_paths:
        path_pom,name = os.path.split(path)
        list_of_file_names.append(name)

    table_list = [table_all[x.replace('.mat','.hea')] for x in list_of_file_names]
    table_array = np.array(table_list)
    table_sum = np.sum(table_array, axis=0)

    print(table_sum)
    return (table_sum)


