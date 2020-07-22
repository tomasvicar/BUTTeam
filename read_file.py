import os
import scipy.io as io
import csv
import pandas
import glob


def read_data(path, file_name):
    data_dict = io.loadmat(os.path.join(path, file_name))
    return data_dict["val"]


class LabelReader:
    def __init__(self, table_path='dx_mapping_scored.csv'):
        self.table = self.read_table(table_path)


    def read_table(self, path):
        """Function reads the table with ALL the diagnosis codes."""

        reader = pandas.read_csv(f'{path}Dx_map.csv')
        del reader['Dx']
        reader.columns = ["Code", "Abb"]
        lbl_code_hash = dict(zip(reader.Code, reader.Abb))

        return lbl_code_hash

    def read_lbl(self, file_name):
        """Function saves information about the patient from header file in this order:
        sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
        SNOMED and abbreviations (sepate lists)"""

        name = os.path.join(file_name) + ".hea"
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'],
                              ['427172004', '17338001']]

        # Read line 15 in header file and parse string with labels
        with open(name, "r") as file:
            list_parameters = []
            for line_idx, line in enumerate(file):
                if line_idx == 0:
                    line_sep = line.split()
                    list_parameters = [line_sep[2], line_sep[3]]
                if line_idx == 1:
                    line_sep = line.split()
                    parameter = line_sep[2]
                    list_parameters.append(int(parameter.replace('/mV', '')))
                if line_idx == 13:
                    list_parameters.append(int(line.replace('#Age: ', '').replace('\n', '')))
                if line_idx == 14:
                    list_parameters.append(line.replace('#Sex: ', '').replace('\n', ''))
                if line_idx == 15:
                    diagnosis = line.replace('#Dx: ', '').replace('\n', '')
                    lbs_to_change = [equivalent_classes[0][1], equivalent_classes[1][1], equivalent_classes[2][1]]
                    diagnosis = diagnosis.split(",")
                    for lbl_idx, lbl in enumerate(diagnosis):
                        if lbl == lbs_to_change[0]:
                            diagnosis[lbl_idx] = equivalent_classes[0][0]
                        if lbl == lbs_to_change[1]:
                            diagnosis[lbl_idx] = equivalent_classes[1][0]
                        if lbl == lbs_to_change[2]:
                            diagnosis[lbl_idx] = equivalent_classes[2][0]

                    list_parameters.append(diagnosis)



                    break
            print(list_parameters)
        file.close()

        list_parameters = self.code_map(self.table, list_parameters)
        print(list_parameters)
        return list_parameters

    def code_map(self, lbl_code_hash, list_parameters):
        """This fucntion maps the numerical codes of diagnosis to abbreviations."""

        test_list = [int(i) for i in list_parameters[-1]]
        codes = [lbl_code_hash[x] for x in test_list]
        list_parameters.append(codes)
        print(list_parameters)

        return list_parameters

path_table = "/Users/petranovotna/Library/Mobile Documents/com~apple~CloudDocs/PhD/2020_CinC_challenge/BUTTeam-master/"
labelReader = LabelReader(path_table)
path_name = "/Users/petranovotna/PycharmProjects/data_challenge_2/Training_2/Q0001"
path_name_2 = "/Users/petranovotna/PycharmProjects/data_challenge_2/Training_2/"

a = labelReader.read_lbl(path_name)


file_list = glob.glob(path_name_2 + r"/*.hea", recursive=True)
print(file_list)
