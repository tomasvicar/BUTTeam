#!/usr/bin/env python

import os, sys
from train_12ECG_classifier import train_12ECG_classifier
from config import Config

if __name__ == '__main__':
    # Parse arguments.
    input_directory = '../data'
    output_directory = '42'

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Running training code...')

    train_12ECG_classifier(input_directory, output_directory)

    print('Done.')
