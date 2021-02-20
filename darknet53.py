from convert_cfg import *


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = model_initialization(self.blocks)

    def forward(self, x, CUDA=True):
        # block[0] contains the [net] details
        modules = self.blocks[1:]
        #We cache the outputs for the route layer
        outputs = {}   
        write = 0     
        for i, module in enumerate(modules):        
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                if len(layers) > 1:
                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]
                    x = torch.cat((map1,map2),1)
                outputs[i] = x
                                
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                # just adding outputs for residual network
                x = outputs[i-1] + outputs[i+from_]  
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                            
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])
                #Get the number of classes
                num_classes = int(module["classes"])
                        
                #Transform 
                x = x.data   # get the data at that point
                x = self.detection_preprocess(x,inp_dim,anchors,num_classes)

                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
                else:       
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i-1]
        try:
            return detections   #return detections if present
        except:
            return 0

    @staticmethod
    def detection_preprocess(x,inp_dim,anchors,num_classes,CUDA=True):
        """
        This function will take input_dimension_of_image,anchors and number of classes as input 
        """
        # x --> 4D feature map
        batch_size = x.size(0)
        grid_size = x.size(2)   # grid_size = inp_dim // stride
        stride =  inp_dim // x.size(2)  # factor by which current feature map reduced from input
    
        # output of each anchor box
        bbox_attrs = 5 + num_classes    # 5 + 80 || x,y,w,h,confidence score + 80
        num_anchors = len(anchors)  # 3 || number of anchors per grid
    
        # 4D to 3D conversion
        # eg detection input dimension [1, 255, 13, 13]
        prediction = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size) # 1x255x169     
        prediction = prediction.transpose(1,2).contiguous() #1x169x255
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs) # output 1x507x85

        # the dimension of anchors is wrt original image. We will make it corresponding to feature map
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
        #Add the center offsets
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1) #(1,gridsize*gridsize,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
        prediction[:,:,:2] += x_y_offset

        #log space transform height and the width
        anchors = torch.FloatTensor(anchors)

        if CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors #width and height
        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))    
        prediction[:,:,:4] *= stride    
        return prediction

    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        # header = torch.from_numpy(header)
        # self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                # we are loading weights as common beacuse when batchnormalization is present there is no bias for conv layer
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # Note: we dont have bias for conv when batch normalization is there