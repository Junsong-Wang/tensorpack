import numpy as np
import time
import math

MIN = -100000
BN_WIDTH = 16
BN_SCALE_Q = 18
BN_BIAS_Q = 10
BN_FIX_POINT_MAX = 2 ** (BN_WIDTH - 1) - 1
BN_FIX_POINT_MIN = - 2 ** (BN_WIDTH - 1)

def data_dump(file_name, mat):
    mat = mat.reshape(-1)
    data_str = ''
    for data in mat:
        data_str += '%f\n'%(data)

    fd = open(file_name, 'w')
    fd.write(data_str)
    fd.close()

def active(x):
    k = 4
    clip_x = min(max(x, 0), float((2**(k-1)-1))/float(2**(k-1)))
    n = float(2**(k-1))
    return math.floor(clip_x * n)

def binarize_weights(weights):
    E = np.mean(np.abs(weights))
    weights[np.where(weights > 0)] = 1
    weights[np.where(weights < 0)] = -1
    weights_binary = weights.astype(np.int32)
    return E, weights_binary

def bn_init(bn, E, auto_q=False, scale_q=BN_SCALE_Q, bias_q=BN_BIAS_Q):
    bn_scale = bn[2] / np.sqrt(bn[1] + 1e-4) * E
    bn_bias = bn[3] - bn[0] * bn[2] / np.sqrt(bn[1] + 1e-4)

    if auto_q is True:
        scale_q = int(math.floor(math.log(1.0/np.max(np.abs(bn_scale))) / math.log(2.0))) + 15
        bias_q = int(math.floor(math.log(1.0/np.max(np.abs(bn_bias))) / math.log(2.0))) + 15
    print 'SCALE_Q:{}, BIAS_Q:{}, SCALE_MAX:{}, BIAS:{}'\
        .format(scale_q, bias_q, np.max(np.abs(bn_scale)), np.max(np.abs(bn_bias)))
       
    bn_scale_fix = bn_scale * (2 ** scale_q)
    bn_scale_fix[np.where(bn_scale_fix > BN_FIX_POINT_MAX)] = BN_FIX_POINT_MAX
    bn_scale_fix[np.where(bn_scale_fix < BN_FIX_POINT_MIN)] = BN_FIX_POINT_MIN
    bn_scale_fix = np.round(bn_scale_fix).astype(np.int32)

    bn_bias_fix = bn_bias * (2 ** bias_q)
    bn_bias_fix[np.where(bn_bias_fix > BN_FIX_POINT_MAX)] = BN_FIX_POINT_MAX
    bn_bias_fix[np.where(bn_bias_fix < BN_FIX_POINT_MIN)] = BN_FIX_POINT_MIN
    bn_bias_fix = np.round(bn_bias_fix).astype(np.int32)
    
    return bn_scale, bn_bias, bn_scale_fix, bn_bias_fix, scale_q, bias_q

def bn_process(data, bn_scale, bn_bias, bn_scale_fix, bn_bias_fix, scale_q, bias_q, idq, odq):
    data_bn_fix = (data * bn_scale_fix + bn_bias_fix * (2**(idq + scale_q - bias_q)))/(2**(idq + scale_q - odq))
    data_bn_real = data / float(2**idq) * bn_scale + bn_bias
    return data_bn_real, data_bn_fix

def conv(input_data, weights, output_shape, stride, pad, bn, idq, odq):    
    E, weights = binarize_weights(weights)
    bn_scale, bn_bias, bn_scale_fix, bn_bias_fix, scale_q, bias_q = bn_init(bn, E, True)  

    input_shape = input_data.shape
    output_data = np.zeros(output_shape)
    output_data_fix = np.zeros(output_shape)
    (kernel_num, kernel_width, kernel_height, kernel_channel) = weights.shape
    for width in range(output_shape[0]):
        for height in range(output_shape[1]):
            for kernel in range(kernel_num):
                conv_sum = 0
                for conv_width in range(kernel_width):
                    for conv_height in range(kernel_height):
                        for conv_channel in range(kernel_channel):
                            channel_index = conv_channel
                            height_index = stride * height - pad + conv_height
                            width_index = stride * width - pad + conv_width
                            data_  = 0 if (height_index < 0 or height_index >= input_shape[1] \
                                           or width_index < 0 or width_index >= input_shape[0]) \
                                       else input_data[width_index, height_index, channel_index]
                            weight_ = weights[kernel, conv_width, conv_height, conv_channel]
                            conv_sum += data_ * weight_

                            conv_bn_real, conv_bn_fix = bn_process(conv_sum, bn_scale[kernel], bn_bias[kernel], bn_scale_fix[kernel],\
                                       bn_bias_fix[kernel], scale_q, bias_q, idq, odq)

                            active_real = active(conv_bn_real)
                            active_fix = min(max(0, conv_bn_fix), 2**odq-1)
                            """
                            weight_display = 1 if weight_== -1 else 0
                            print 'data:(', width_index, height_index, channel_index, ') ', data_, \
                                  'weight:(', conv_width, conv_height, conv_channel, ') ', weight_display, \
                                  'conv_sum:', conv_sum
                            """
                """
                print 'Position:', width, height, kernel, \
                      'real_conv:', conv_sum/float(2**idq) * E, 'real_bn:', conv_sum_real, 'real_active:', active(conv_sum_real), \
                      'sim_conv:', conv_sum, 'bn_param', bn_scale_fix[kernel], bn_bias_fix[kernel], \
                      'sim_bn0:', conv_sum_biased, 'sim_active:', active_sim
                """
                output_data[width, height, kernel] = active_real
                output_data_fix[width, height, kernel] = active_fix
    return output_data, output_data_fix


def pool(input_data, output_shape, kernel_size, stride, pad):
    input_shape = input_data.shape
    output_data = np.zeros(output_shape)
    for width in range(output_shape[0]):
        for height in range(output_shape[1]):
            for kernel in range(output_shape[2]):
                pool_max = MIN
                for pool_width in range(kernel_size):
                    for pool_height in range(kernel_size):
                        height_index = stride * height - pad + pool_height
                        width_index = stride * width - pad + pool_width
                        data_  = MIN if (height_index < 0 or height_index >= input_shape[1] \
                                           or width_index < 0 or width_index >= input_shape[0]) \
                                       else input_data[width_index, height_index, kernel]
                        pool_max = data_ if data_ > pool_max else pool_max
                output_data[width, height, kernel] = pool_max
                
    return output_data

def fc(input_data, weights, bn, idq, odq, isactive=True):
    print input_data.shape, weights.shape
    print bn[0].shape
    kernel_num = weights.shape[0]
    if input_data.ndim == 3:
        #input_data = input_data.transpose(1,0,2)
        width, height, channel = input_data.shape
        weights = weights.reshape(kernel_num, height, width, channel).transpose(0, 2, 1, 3).reshape(kernel_num, -1)
    input_data = input_data.reshape(-1)

    E, weights = binarize_weights(weights)
    print E
    if bn:
        bn_scale, bn_bias, bn_scale_fix, bn_bias_fix, scale_q, bias_q= bn_init(bn, E, True)

    output_data = np.zeros(kernel_num, dtype=np.float32)
    output_data_fix = np.zeros(kernel_num)
    for kernel in range(kernel_num):
        dot_value = np.dot(input_data, weights[kernel,:])
        if bn:
            dot_bn_real, dot_bn_fix = bn_process(dot_value, bn_scale[kernel], bn_bias[kernel], bn_scale_fix[kernel],\
                                       bn_bias_fix[kernel], scale_q, bias_q, idq, odq)
        else:
            dot_bn_fix = dot_value
            dot_bn_real = dot_value_real = dot_value / float(2**idq) * E
            
        if isactive:
            active_fix = min(max(0, dot_bn_fix), 2**odq-1)
            active_real = active(dot_bn_real)
        else:
            active_fix = dot_bn_fix
            active_real = dot_bn_real

        output_data[kernel] = active_real
        #output_data[kernel] = dot_value  * E / 1.0165
        output_data_fix[kernel] = active_fix

        #print '---------', kernel, dot_value, '----------------'
        #print bn_scale[kernel], bn_bias[kernel], dot_bn_real, active_real
        #print bn_scale_fix[kernel], bn_bias_fix[kernel], dot_bn_fix / 8.0, active_fix

    return output_data, output_data_fix

