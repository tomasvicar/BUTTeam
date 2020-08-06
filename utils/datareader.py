import os
import scipy.io as io
import pandas
import math
import glob

__all__ = ["DataReader"]


class DataReader:
    """Data manipulation class wrapper"""

    # Remap snomed code duplicates
    snomed_mapping = {
        "59118001": "713427006",
        "63593006": "284470004",
        "17338001": "427172004",
    }

    # Remap sex categories description
    sex_mapping = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male",
    }

    @staticmethod
    def read_table(path="tables/"):
        """Function reads the table with ALL the diagnosis codes."""
        table = pandas.read_csv(f'{path}Dx_map.csv', usecols=[1, 2])
        table.columns = ["Code", "Label"]
        return dict(zip(table.Code, table.Label))

    @staticmethod
    def read_sample(file_name):
        """Reads mat data as np array"""
        if os.path.exists(file_name):
            return io.loadmat(os.path.join(file_name))["val"]
        else:
            return None

    @staticmethod
    def read_header(file_name, snomed_table,from_file=True):
        """Function saves information about the patient from header file in this order:
        sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
        SNOMED and abbreviations (sepate lists)"""

        sampling_frequency, resolution, age, sex, snomed_codes, labels = [], [], [], [], [], []

        def string_to_float(input_string):
            """Converts string to floating point number"""
            try:
                value = float(input_string)
            except ValueError:
                value = None

            if math.isnan(value):
                return None
            else:
                return value

        if from_file:
            lines=[]
            with open(file_name, "r") as file:
                for line_idx, line in enumerate(file):
                    lines.append(line)
        else:
            lines=file_name

        # Read line 15 in header file and parse string with labels

        snomed_codes = []
        resolution=[]
        age=None
        sex=None
        for line_idx, line in enumerate(lines):
            if line_idx == 0:
                sampling_frequency = float(line.split(" ")[2])
                continue
            if 1<=line_idx<=12:
                resolution.append(string_to_float(line.split(" ")[2].replace("/mV", "").replace("/mv", "")))
                continue
            if line.startswith('#Age'):
                age = string_to_float(line.replace("#Age:","").replace("#Age","").rstrip("\n").strip())
                continue
            if line.startswith('#Sex'):
                sex = line.replace("#Sex:","").replace("#Sex","").rstrip("\n").strip().lower()
                if sex not in DataReader.sex_mapping:
                    sex = None
                else:
                    sex = DataReader.sex_mapping[sex]
                continue
            if line.startswith('#Dx'):
                if from_file:
                    snomed_codes = line.replace("#Dx:","").replace("#Dx","").rstrip("\n").strip().split(",")
                    snomed_codes = [DataReader.snomed_mapping.get(item, item) for item in snomed_codes]
                continue


        return sampling_frequency, resolution, age, sex, snomed_codes

    @staticmethod
    def get_label_maps(path="tables/"):
        """Function reads the table with ALL the diagnosis codes."""

        reader = pandas.read_csv(f'{path}dx_mapping_scored.csv', usecols=[1, 2])
        reader.columns = ["Code", "Labels"]

        snomed_codes, labels = reader.Code, reader.Labels

        label_mapping = {str(code): label for code, label in zip(snomed_codes, labels)
                         if str(code) not in DataReader.snomed_mapping}

        idx_mapping = {key: idx for idx, key in enumerate(label_mapping)}

        return idx_mapping, label_mapping


def main():
    """Test function"""
    input_directory = r"E:\data\Physionet2020"
    file_list = glob.glob(input_directory + r"\**\*.hea", recursive=True)

    sample_file_name = "E:\\data\\Physionet2020\\Training_StPetersburg\\I0043.mat"
    header_file_name = "E:\\data\\Physionet2020\\Training_StPetersburg\\I0043.hea"
    snomed_table = DataReader.read_table(path="")

    sample = DataReader.read_sample(sample_file_name)
    header = DataReader.read_header(header_file_name, snomed_table)
    idx_mapping, label_mapping = DataReader.get_label_maps(path="")


if __name__ == "__main__":
    main()

