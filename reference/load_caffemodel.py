import numpy as np
import scipy.io as sio
import caffe

def load():
    # Load the net
    caffe.set_mode_cpu()
    # You may need to train this caffemodel first
    # There should be script to help you do the training
    net = caffe.Net(root + 'train_val.prototxt', root + 'fish-cvr_iter_20000.caffemodel',\
        caffe.TEST)
    layer_names = ["ifc0", "ifc1", "ifc2", "ifc3",
                    "ufc0", "ufc1", "ufc2", "ufc3",
                    "qfc0", "qfc1", "qfc2", "qfc3",
                    "fcc", "fcc0", "fcc1", "fcc2"]
    for layer in layer_names:
        weight = net.params[layer][0].data
        bias = net.params[layer][1].data

        print "="*30 + layer + "weight" + "="*30
        print "="*30 + layer + "bias" + "="*30

    # ifc0_w = net.params['ifc0'][0].data
    # ifc0_b = net.params['ifc0'][1].data
    # conv2_w = net.params['conv2'][0].data
    # conv2_b = net.params['conv2'][1].data
    # ip1_w = net.params['ip1'][0].data
    # ip1_b = net.params['ip1'][1].data
    # ip2_w = net.params['ip2'][0].data
    # ip2_b = net.params['ip2'][1].data

    # print "="*20 + "conv1_w" + "="*20
    # print ifc0_w
    #
    # print "="*20 + "conv1_w" + "="*20
    # print ifc0_b


if __name__ == "__main__":
    # You will need to change this path
    root = '/home/zhimo.bmz/DeepCTR-master/examples/fish-dnn/'
    load()
    print 'Caffemodel loaded and written to .mat files successfully!'