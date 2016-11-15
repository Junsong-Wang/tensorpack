import numpy as np
import tensorflow as tf
import pickle

def compute_mean(data):
    numpy_mean = np.mean(np.abs(data))

    vec = tf.placeholder(tf.float32, data.shape)
    avg = tf.reduce_mean(tf.abs(vec))
    
    with tf.Session() as sess:
        tf_mean = sess.run(avg, feed_dict={vec: data})

    return numpy_mean, tf_mean 
    
if __name__ == '__main__':

    weight_file = 'zf_pad_64.npy'
    param_dict = np.load(weight_file).item()

    weights_name = ['conv0/W:0', 'conv1/W:0', 'conv2/W:0', 'conv3/W:0', 'conv4/W:0', 'fc0/W:0', 'fc1/W:0', 'fct/W:0']

    compensation_factor = {}
    for layer in weights_name:
        data = param_dict[layer]
        numpy_mean, tf_mean = compute_mean(data)
        compensation_factor[layer] = numpy_mean/tf_mean
        print 'Layer:{}\tnumpy:{}\ttensflow:{}\tratio:{}'\
               .format(layer, numpy_mean, tf_mean, numpy_mean/tf_mean)
    with open('compensation.pkl', 'wb') as f:
        pickle.dump(compensation_factor, f)
