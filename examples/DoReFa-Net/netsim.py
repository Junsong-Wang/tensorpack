import numpy as np
import time
import math

from cnn_lib import conv, pool, data_dump

if __name__ == '__main__':
    data_file = './dump/input.dat'
    weight_file = 'zf_small.npy'
    input_shape = (224, 224, 3)
    output_shape = (112, 112, 16)

    sim_fix = True

    data_list = []
    fp = open(data_file, 'r')
    for line in fp:
        data_list.append(int(line.strip()))
    input_data = np.array(data_list).reshape(input_shape)
    
    param_dict = np.load(weight_file).item()
       

    # start to simulation
    print 'simulate conv0...'
    weights = param_dict['conv0/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn0/mean/EMA:0'], param_dict['bn0/variance/EMA:0'], param_dict['bn0/gamma:0'], param_dict['bn0/beta:0']]
    conv0_output, conv0_output_fix = conv(input_data.astype(np.int32), weights, (112, 112, 16), 2, 2, bn, 7, 3)
    if sim_fix:
        data_dump('./sim_fix/conv0.data', conv0_output_fix/8.0)
    else:
        data_dump('./sim/conv0.data', conv0_output/8.0)

    print 'simulate pool0...'
    if sim_fix:
        pool0_output = pool(conv0_output_fix.astype(np.int32), (56,56,16), 3, 2, 0)
        data_dump('./sim_fix/pool0.data', pool0_output/8.0)
    else:
       pool0_output = pool(conv0_output.astype(np.int32), (56,56,16), 3, 2, 0)
       data_dump('./sim/pool0.data', pool0_output/8.0)


    print 'simulate conv1...'
    weights = param_dict['conv1/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn1/mean/EMA:0'], param_dict['bn1/variance/EMA:0'], param_dict['bn1/gamma:0'], param_dict['bn1/beta:0']]
    conv1_output, conv1_output_fix= conv(pool0_output.astype(np.int32), weights, (28, 28, 64), 2, 1, bn, 3, 3)
    if sim_fix:
       data_dump('./sim_fix/conv1.data', conv1_output_fix/8.0)
    else:
        data_dump('./sim/conv1.data', conv1_output/8.0)

    print 'simulate pool1...'
    if sim_fix:
        pool1_output = pool(conv1_output_fix.astype(np.int32), (14,14,64), 3, 2, 0)
        data_dump('./sim_fix/pool1.data', pool1_output/8.0)
    else:
        pool1_output = pool(conv1_output.astype(np.int32), (14,14,64), 3, 2, 0)
        data_dump('./sim/pool1.data', pool1_output/8.0)

    print 'simulate conv2...'
    weights = param_dict['conv2/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn2/mean/EMA:0'], param_dict['bn2/variance/EMA:0'], param_dict['bn2/gamma:0'], param_dict['bn2/beta:0']]
    conv2_output, conv2_output_fix= conv(pool1_output.astype(np.int32), weights, (14, 14, 128), 1, 1, bn, 3, 3)
    if sim_fix:
        data_dump('./sim_fix/conv2.data', conv2_output_fix/8.0)
    else:
        data_dump('./sim/conv2.data', conv2_output/8.0)

    print 'simulate conv3...'
    weights = param_dict['conv3/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn3/mean/EMA:0'], param_dict['bn3/variance/EMA:0'], param_dict['bn3/gamma:0'], param_dict['bn3/beta:0']]
    if sim_fix:
       conv3_output, conv3_output_fix= conv(conv2_output_fix.astype(np.int32), weights, (14, 14, 128), 1, 1, bn, 3, 3)
       data_dump('./sim_fix/conv3.data', conv3_output_fix/8.0)
    else:
       conv3_output, conv3_output_fix= conv(conv2_output.astype(np.int32), weights, (14, 14, 128), 1, 1, bn, 3, 3)
       data_dump('./sim/conv3.data', conv3_output/8.0)

    print 'simulate conv4...'
    weights = param_dict['conv4/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn4/mean/EMA:0'], param_dict['bn4/variance/EMA:0'], param_dict['bn4/gamma:0'], param_dict['bn4/beta:0']]
    if sim_fix:
       conv4_output, conv4_output_fix= conv(conv3_output_fix.astype(np.int32), weights, (14, 14, 64), 1, 1, bn, 3, 3)
       data_dump('./sim_fix/conv4.data', conv4_output_fix/8.0)
    else:
       conv4_output, conv4_output_fix= conv(conv3_output.astype(np.int32), weights, (14, 14, 64), 1, 1, bn, 3, 3)
       data_dump('./sim/conv4.data', conv4_output/8.0)

    print 'simulate pool4...'
    if sim_fix:
        pool4_output = pool(conv4_output_fix.astype(np.int32), (7,7,64), 3, 2, 0)
        data_dump('./sim_fix/pool4.data', pool4_output/8.0)
    else:
        pool4_output = pool(conv4_output.astype(np.int32), (7,7,64), 3, 2, 0)
        data_dump('./sim/pool4.data', pool4_output/8.0)

