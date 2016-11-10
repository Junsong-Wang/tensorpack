import numpy as np
import time
import math

from cnn_lib import conv, pool, data_dump

if __name__ == '__main__':
    data_file = './dump/input.dat'
    weight_file = 'zf_pad_64.npy'
    input_shape = (224, 224, 4)
    output_shape = (112, 112, 64)

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
    conv0_output, conv0_output_fix = conv(input_data.astype(np.int32), weights, (112, 112, 64), 2, 2, bn, 7, 3)
    if sim_fix:
        data_dump('./sim_fix/conv0.dat', conv0_output_fix/8.0)
    else:
        data_dump('./sim/conv0.dat', conv0_output/8.0)

    print 'simulate pool0...'
    if sim_fix:
        pool0_output = pool(conv0_output_fix.astype(np.int32), (56,56,64), 3, 2, 0)
        data_dump('./sim_fix/pool0.dat', pool0_output/8.0)
    else:
       pool0_output = pool(conv0_output.astype(np.int32), (56,56,64), 3, 2, 0)
       data_dump('./sim/pool0.dat', pool0_output/8.0)

    
    print 'simulate conv1...'
    weights = param_dict['conv1/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn1/mean/EMA:0'], param_dict['bn1/variance/EMA:0'], param_dict['bn1/gamma:0'], param_dict['bn1/beta:0']]
    conv1_output, conv1_output_fix= conv(pool0_output.astype(np.int32), weights, (28, 28, 256), 2, 1, bn, 3, 3)
    if sim_fix:
       data_dump('./sim_fix/conv1.dat', conv1_output_fix/8.0)
    else:
        data_dump('./sim/conv1.dat', conv1_output/8.0)

    print 'simulate pool1...'
    if sim_fix:
        pool1_output = pool(conv1_output_fix.astype(np.int32), (14,14,256), 3, 2, 0)
        data_dump('./sim_fix/pool1.dat', pool1_output/8.0)
    else:
        pool1_output = pool(conv1_output.astype(np.int32), (14,14,256), 3, 2, 0)
        data_dump('./sim/pool1.dat', pool1_output/8.0)

    print 'simulate conv2...'
    weights = param_dict['conv2/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn2/mean/EMA:0'], param_dict['bn2/variance/EMA:0'], param_dict['bn2/gamma:0'], param_dict['bn2/beta:0']]
    conv2_output, conv2_output_fix= conv(pool1_output.astype(np.int32), weights, (14, 14, 384), 1, 1, bn, 3, 3)
    if sim_fix:
        data_dump('./sim_fix/conv2.dat', conv2_output_fix/8.0)
    else:
        data_dump('./sim/conv2.dat', conv2_output/8.0)

    print 'simulate conv3...'
    weights = param_dict['conv3/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn3/mean/EMA:0'], param_dict['bn3/variance/EMA:0'], param_dict['bn3/gamma:0'], param_dict['bn3/beta:0']]
    if sim_fix:
       conv3_output, conv3_output_fix= conv(conv2_output_fix.astype(np.int32), weights, (14, 14, 384), 1, 1, bn, 3, 3)
       data_dump('./sim_fix/conv3.dat', conv3_output_fix/8.0)
    else:
       conv3_output, conv3_output_fix= conv(conv2_output.astype(np.int32), weights, (14, 14, 384), 1, 1, bn, 3, 3)
       data_dump('./sim/conv3.dat', conv3_output/8.0)

    print 'simulate conv4...'
    weights = param_dict['conv4/W:0'].transpose(3,1,0,2)
    bn = [param_dict['bn4/mean/EMA:0'], param_dict['bn4/variance/EMA:0'], param_dict['bn4/gamma:0'], param_dict['bn4/beta:0']]
    if sim_fix:
       conv4_output, conv4_output_fix= conv(conv3_output_fix.astype(np.int32), weights, (14, 14, 256), 1, 1, bn, 3, 3)
       data_dump('./sim_fix/conv4.dat', conv4_output_fix/8.0)
    else:
       conv4_output, conv4_output_fix= conv(conv3_output.astype(np.int32), weights, (14, 14, 256), 1, 1, bn, 3, 3)
       data_dump('./sim/conv4.dat', conv4_output/8.0)

    print 'simulate pool4...'
    if sim_fix:
        pool4_output = pool(conv4_output_fix.astype(np.int32), (7,7,256), 3, 2, 0)
        data_dump('./sim_fix/pool4.dat', pool4_output/8.0)
    else:
        pool4_output = pool(conv4_output.astype(np.int32), (7,7,256), 3, 2, 0)
        data_dump('./sim/pool4.dat', pool4_output/8.0)

    print 'simulation fc0...'
    weights = param_dict['fc0/W:0'].transpose(1,0)
    bn = [param_dict['bnfc0/mean/EMA:0'], param_dict['bnfc0/variance/EMA:0'], param_dict['bnfc0/gamma:0'], param_dict['bnfc0/beta:0']]
    fc0_output, fc0_output_fix = fc(poo4_output.astype(np.int32), weights, bn, 3, 3, True)
    if sim_fix:
        data_dump('./sim_fix/fc0.dat', fc0_output_fix/8.0)
    else:
        data_dump('./sim/fc0.dat', fc0_output/8.0)

    print 'simulation fc1'
    weights = param_dict['fc1/W:0'].transpose(1,0)
    bn = [param_dict['bnfc1/mean/EMA:0'], param_dict['bnfc1/variance/EMA:0'], param_dict['bnfc1/gamma:0'], param_dict['bnfc1/beta:0']]
    if sim_fix:
        fc1_output, fc1_output_fix = fc(fc0_output_fix.astype(np.int32), weights, bn, 3, 3, True)
        data_dump('./sim_fix/fc0.dat', fc1_output_fix/8.0)
    else:
        fc1_output, fc1_output_fix = fc(fc0_output.astype(np.int32), weights, bn, 3, 3, True)
        data_dump('./sim/fc0.dat', fc0_output/8.0)

    print 'simulation fct'
    weights = param_dict['fct/W:0'].transpose(1,0)
    if sim_fix:
        fct_output, fct_output_fix = fc(fc1_output_fix.astype(np.int32), weights, None, 3, 3, False)
        data_dump('./sim_fix/fct.dat', fct_output_fix/8.0)
        data_dump('./sim_fix/final.dat', fct_output_fix)
    else:
        fct_output, fct_output_fix = fc(fc1_output.astype(np.int32), weights, None, 3, 3, False)
        data_dump('./sim/fct.dat', fct_output/8.0)


