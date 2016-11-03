#!/usr/bin/env python
"""
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import copy

def read_data(data_file):
    data_list = []
    fp = open(data_file, 'r')
    for line in fp:
        data_list.append(float(line.strip()))
    return np.array(data_list)


def compare(file1, file2):
    
    data1 = read_data(file1)
    data2 = read_data(file2) 

    diff  = data1 - data2
 
    unequal_num = len(np.where(diff != 0)[0])
    print unequal_num, float(unequal_num) / float(len(data1))
    
    plt.figure(1)
    plot1 = plt.plot(np.arange(len(data1)), diff, 'r')
    plt.show()


if __name__ == '__main__':

    compare(sys.argv[1], sys.argv[2])

