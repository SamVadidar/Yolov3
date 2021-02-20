from darknet53 import *
from nms import *

# function to load the classes
def load_classes(class_file):
    fp = open(class_file, "r")
    names = fp.read().split("\n")[:-1]
    return names

# function converting images from opencv format to torch format
def preprocess_image(img, inp_dim, CUDA, mode = 'video'):
    """
    Prepare image for inputting to the neural network. 
    
    Returns processed image, original image and original image dimension  
    """
    if mode == 'image':
        orig_im = cv2.imread(img)
    else:
        orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (canvas_image(orig_im, (inp_dim, inp_dim)))
    img = img[:,:,::-1]
    img_ = img.transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    if CUDA:
        img_ = img_.cuda()
    return img_, orig_im, dim

#function letterbox_image that resizes our image, keeping the 
# aspect ratio consistent, and padding the left out areas with the color (128,128,128)
def canvas_image(img, conf_inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = conf_inp_dim  # dimension from configuration file

    ratio = min(w/img_w, h/img_h)

    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    # we fill the extra pixels with 128
    canvas = np.full((conf_inp_dim[1], conf_inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image
    
    return canvas

def draw_boxes(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0,0,255)
    cv2.rectangle(img, c1, c2,color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

##############################################################################################
cap = cv2.VideoCapture(0)

CUDA = torch.cuda.is_available()
#frame = "./test_photo/dog.jpg"
weightsfile = "./weights/yolov3.weights"
classfile = "./data/coco.names"
nms_thesh = 0.5
#Set up the neural network
print("Loading network.....")
model = Darknet(config_file)
model.load_weights(weightsfile)
print("Network successfully loaded")
classes = load_classes(classfile)
print('Classes loaded')


conf_inp_dim = int(model.net_info["height"])#608
while(True):

    retBool, frame = cap.read()

    # treading and resizing image
    processed_image, original_image, original_img_dim = preprocess_image(frame,conf_inp_dim, CUDA)
    #print(processed_image.shape)

    im_dim = original_img_dim[0], original_img_dim[1]
    im_dim = torch.FloatTensor(im_dim).repeat(1,2)

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        # im_dim = im_dim_list.cuda()
        im_dim = im_dim.cuda()
        model.cuda()

    #Set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model(processed_image)


    output = final_detection(prediction, confidence_threshold=0.5, num_classes=80, nms_conf = nms_thesh)

    im_dim_list = torch.index_select(im_dim, 0, output[:,0].long())

    scaling_factor = torch.min(conf_inp_dim/im_dim_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor

    # adjusting bounding box size between 0 and configuration image size
    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(conf_inp_dim))

    list(map(lambda x: draw_boxes(x, original_image), output))
    cv2.imshow("Yolov3_frame", original_image)
    # cv2.imwrite("output.png", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()