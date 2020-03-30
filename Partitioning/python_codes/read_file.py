import os
import json
import scipy.io as io


def read_header(path, file_name):
    """Read and return class labels from CinC header file"""
    labels = []

    # Read line 15 in header file and parse string with labels
    with open(os.path.join(path, file_name), "r") as file:
        for line_idx, line in enumerate(file):
            if line_idx == 15:
                line = line.rstrip("\n")
                line = line.strip("#Dx: ")
                labels.append(line.split(","))
                break
    file.close()
    return labels


def read_data(path, file_name):
    data_dict = io.loadmat(os.path.join(path, file_name))
    return data_dict["val"]


def get_labels(path):
    keys = []
    labels = []

    for file_name in os.listdir(path):
        if file_name.endswith(".hea"):
            keys.append(file_name.rstrip(".hea"))
            labels.extend(read_header(path, file_name))
    return dict(zip(keys, labels))


def json_save(data, file_name):
    with open(file_name, "w") as file:
        json.dump(data, file)


def main():
    path = "../data/cinc2020/"
    label_file_name = "labels.json"

    labels = get_labels(path)
    json_save(labels, label_file_name)


if __name__ == "__main__":
    main()
