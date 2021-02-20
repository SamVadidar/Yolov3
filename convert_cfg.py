import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import os
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt


config_file = './config/yolov3.cfg'


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(config_file):
    file = open(config_file,'r')
    file = file.read().split('\n')
    file = [line for line in file if len(line)>0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    final_list = []
    element_dict = {}
    for line in file:

        if line[0] == '[':
            if len(element_dict) != 0:     # appending the dict stored on previous iteration
                final_list.append(element_dict)
                element_dict = {} # again emtying dict
            element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
            
        else:
            val = line.split('=')
            element_dict[val[0].rstrip()] = val[1].lstrip()  #removing spaces on left and right side
        
    final_list.append(element_dict) # appending the values stored for last set
    return final_list

def model_initialization(blocks):
    darknet_details = blocks[0]
    channels = 3 
    output_filters = []  #list of filter numbers in each layer.It is useful while defining number of filters in routing layer
    modulelist = nn.ModuleList()

    for i,block in enumerate(blocks[1:]):
        seq = nn.Sequential()

        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            use_bias= False if ("batch_normalize" in block) else True
            pad = (kernel_size - 1) // 2
            conv = nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size, 
                                stride=strides, padding=pad, bias = use_bias)
            
            seq.add_module("conv_{0}".format(i), conv)
            
            if "batch_normalize" in block:
                bn = nn.BatchNorm2d(filters)
                seq.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                seq.add_module("leaky_{0}".format(i), activn)
                
        elif (block["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            seq.add_module("upsample_{}".format(i), upsample)

        elif (block["type"] == 'route'):
            # start and end is given in format (eg:-1 36 so we will find layer number from it.
            # we will find layer number in negative format
            # so that we can get the number of filters in that layer
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])
            if len(block['layers']) == 1:               
                #ie if -1 given and present layer is 20 . we have to sum filters in 19th and 20th layer 
                block['layers'][0] = int(i + start)             
                filters = output_filters[block['layers'][0]]  #start layer number
            elif len(block['layers']) > 1:
                # suppose we have -1,28 and present layer is 20 we have sum filters in 19th and 28th layer
                block['layers'][0] = int(i + start) 
                # block['layers'][1] = int(block['layers'][1]) - i # end layer number  
                block['layers'][1] = int(block['layers'][1])
                filters = output_filters[block['layers'][0]] + output_filters[block['layers'][1]]
                    
            # that means this layer don't have any forward operation
            route = DummyLayer()
            seq.add_module("route_{0}".format(i),route)
                    
        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            shortcut = DummyLayer()
            seq.add_module("shortcut_{0}".format(i),shortcut)

        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(",")
            anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            block["anchors"] = anchors
                    
            detectorLayer = DetectionLayer(anchors)
            seq.add_module("Detection_{0}".format(i),detectorLayer)

        modulelist.append(seq)
        output_filters.append(filters)     
        channels = filters

    return darknet_details, modulelist


if __name__ == "__main__":
    config_file = './config/yolov3.cfg'
    blocks =  parse_cfg(config_file)
    details,modules = model_initialization(blocks)
    print(modules)
