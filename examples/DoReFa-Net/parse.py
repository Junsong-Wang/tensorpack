import numpy as np
import six
param_dict = np.load('alexnet-126.npy').item()
#print param_dict['conv1/W'].transpose(3,2,0,1).shape
for n, v in six.iteritems(param_dict):
    print n, v.shape
