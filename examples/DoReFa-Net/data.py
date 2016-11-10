#!/usr/bin/env python
"""
"""
import numpy as np
import struct

DW = 8
FIX_POINT_MAX = 2 ** (DW - 1) - 1
FIX_POINT_MIN = - 2 ** (DW - 1)
UINT8_MAX = 2 ** 8

def mat_dump_hex(file_path_name, mat, pad=False):
    blob_data = mat[0]
    blob_data = blob_data.transpose(1, 0, 2)
    if pad:
        blob_data_pad = np.zeros((blob_data.shape[0], blob_data.shape[1], 4))
        blob_data_pad[:,:,0:3] = blob_data
        blob_data_shaped = blob_data_pad.reshape(-1)
    else:
        blob_data_shaped = blob_data.reshape(-1)
    if len(blob_data_shaped) % 64:
        blob_data_shaped = np.append(blob_data_shaped, \
                    np.zeros(64 - len(blob_data_shaped) % 64, dtype=int))
    blob_data_shaped = blob_data_shaped.reshape(-1, 64)

    mat = np.round(blob_data_shaped)

    mat[np.where(mat > FIX_POINT_MAX)] = FIX_POINT_MAX
    mat[np.where(mat < FIX_POINT_MIN)] = FIX_POINT_MIN
    #saturate processing
    mat = mat.astype(np.int32)
    #covert to complement format
    mat = (mat + UINT8_MAX) % UINT8_MAX

    coe_str = ''
    for row in mat:
        row_str = ''
        for col in row:
            row_str = '%02X'%(col) + row_str
        row_str += '\n'
        coe_str += row_str
    #write to coe file
    fd = open(file_path_name, 'w')
    fd.write(coe_str)
    fd.close()

def mat_dump_float(file_path_name, mat):
    if mat.ndim == 4:
        print '-----------', mat[0].transpose(1, 0, 2).shape
        mat = mat[0].transpose(1, 0, 2).reshape(-1)
    elif mat.ndim == 2:
        mat = mat[0].reshape(-1)
    else:
        raise Exception ('Mat dim error.')

    mat = mat
    data_str = ''
    for data in mat:
        data_str += '%f\n'%(data)

    fd = open(file_path_name, 'w')
    fd.write(data_str)
    fd.close()

def mat_dump_int(file_path_name, mat, pad=False):
    blob_data = mat[0]
    blob_data = blob_data.transpose(1, 0, 2)
    if pad:
        blob_data_pad = np.zeros((blob_data.shape[0], blob_data.shape[1], 4))
        blob_data_pad[:,:,0:3] = blob_data
        blob_data_shaped = blob_data_pad.reshape(-1)
    else:
        blob_data_shaped = blob_data.reshape(-1)

    mat = blob_data_shaped
    mat[np.where(mat > FIX_POINT_MAX)] = FIX_POINT_MAX
    mat[np.where(mat < FIX_POINT_MIN)] = FIX_POINT_MIN
    #saturate processing
    mat = mat.astype(np.int32)

    data_str = ''
    for data in mat:
        data_str += '%d\n'%(data)

    fd = open(file_path_name, 'w')
    fd.write(data_str)
    fd.close()
