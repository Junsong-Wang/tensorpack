import numpy as np
import time
import math

MIN = -100000
BN_Q = 13
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

def conv(input_data, weights, output_shape, stride, pad, bn, idq, odq):
    E = np.mean(np.abs(weights))
    weights[np.where(weights > 0)] = 1
    weights[np.where(weights < 0)] = -1
    weights = weights.astype(np.int32)

    bn_scale = bn[2] / np.sqrt(bn[1] + 1e-4)
    bn_bias = bn[3] - bn[0] * bn[2] / np.sqrt(bn[1] + 1e-4)

    bn_scale = bn[2] / np.sqrt(bn[1] + 1e-4)
    bn_bias = bn[3] - bn[0] * bn[2] / np.sqrt(bn[1] + 1e-4)

    bn_scale_fix = (bn_scale * E )* (2 ** BN_Q)
    bn_scale_fix = np.round(bn_scale_fix).astype(np.int32)
    bn_bias_fix = bn_bias * (2 ** BN_Q)
    bn_bias_fix = np.round(bn_bias_fix).astype(np.int32)

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

                            conv_sum_scaled = conv_sum * bn_scale_fix[kernel]
                            conv_sum_biased = conv_sum_scaled + bn_bias_fix[kernel] * (2**idq)
                            conv_sum_bn_float = conv_sum_biased / float(((2**(idq + BN_Q))))
                            conv_sum_bn = conv_sum_biased / ((2**(idq + BN_Q - odq)))
                            active_sim = min(max(0, conv_sum_bn), 2**odq-1)

                            conv_sum_real = conv_sum / (2**idq) * E * bn_scale[kernel] + bn_bias[kernel]

                            """
                            print 'data:(', width_index, height_index, channel_index, ') ', data_, \
                                  'weight:(', conv_width, conv_height, conv_channel, ') ', weight_, \
                                  'conv_sum:', conv_sum

                print 'Position:', width, height, kernel, \
                      'real_conv:', conv_sum/dq * E, 'real_bn:', conv_sum / dq.0 * E * bn_scale[kernel] + bn_bias[kernel], \
                      'real_active:', active(conv_sum_real), 'sim_conv:', conv_sum, 'sim_bn0:', conv_sum_bn_float, 'sim_active:', active_sim
                """
                output_data[width, height, kernel] = active(conv_sum_real)
                output_data_fix[width, height, kernel] = active_sim
      

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

