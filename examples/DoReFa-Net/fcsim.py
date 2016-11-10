import numpy as np
import time
import math

from cnn_lib import conv, pool, fc, data_dump

if __name__ == '__main__':
    #data_file = './dump/pool4.dat'
    data_file = './sim_fix-bak/pool4.dat'
    weight_file = 'zf_pad_64.npy'
    input_shape = (7, 7, 256)

    sim_fix = True

    data_list = []
    fp = open(data_file, 'r')
    for line in fp:
        data_list.append(int(float(line.strip()) * 8.0))
    #print data_list
    input_data = np.array(data_list).reshape(input_shape)
    param_dict = np.load(weight_file).item()
       
    # start to simulation
    print 'simulate fc0...'
    weights = param_dict['fc0/W:0'].transpose(1,0)
    bn = [param_dict['bnfc0/mean/EMA:0'], param_dict['bnfc0/variance/EMA:0'], param_dict['bnfc0/gamma:0'], param_dict['bnfc0/beta:0']]
    fc0_output, fc0_output_fix = fc(input_data.astype(np.int32), weights, bn, 3, 3, True)
    if sim_fix:
        data_dump('./sim_fix/fc0.dat', fc0_output_fix/8.0)
    else:
        data_dump('./sim/fc0.dat', fc0_output/8.0)
