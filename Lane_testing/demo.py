import os
from erf_settings import *
import numpy as np
from tools import prob_to_lines as ptl
import cv2
import models
import torch
import torch.nn.functional as F
from options.options import parser
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy
import utils.transforms as tf
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torch.nn as nn
from torch.nn import init

#cap_name = '/home/chen/data/video/test.mp4'
cap_name = 'test/05250340_0277.MP4'
#image = '/home/user/Datasets/culane/driver_100_30frame/05250419_0290.MP4/04080.jpg'
#image = '/home/user/Datasets/culane/driver_100_30frame/05252325_0554.MP4/03630.jpg'  wet
#image = '/home/user/Datasets/culane/driver_193_90frame/06060946_0829.MP4/04950.jpg'  indoor
#image = '/home/user/Datasets/culane/driver_100_30frame/05250340_0277.MP4/00210.jpg'
#image = '/home/user/Datasets/culane/driver_100_30frame/05250343_0278.MP4/00300.jpg'   
#image = '/home/user/Datasets/culane/driver_100_30frame/05250343_0278.MP4/00780.jpg'   
#image = '/home/user/Datasets/culane/driver_100_30frame/05250343_0278.MP4/02910.jpg' 
#image = '/home/user/Datasets/culane/driver_100_30frame/05250358_0283.MP4/00450.jpg'    
#image = '/home/user/Datasets/culane/driver_100_30frame/05250358_0283.MP4/00810.jpg'    
#image = '/home/user/Datasets/culane/driver_100_30frame/05250635_0332.MP4/01470.jpg'   
#image = '/home/user/Datasets/culane/driver_100_30frame/05250635_0332.MP4/01890.jpg'  
image = '/home/user/Datasets/culane/driver_100_30frame/05250343_0278.MP4/05220.jpg'    
file_path = "test8_night.txt"

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)
    # model
    model_A = models.ResG(3,3,'A')
    model_FTFA = models.FTF(3,3)
    model_erf = models.ERFNet3(5)
    input_mean = model_erf.input_mean
    input_std = model_erf.input_std
    model_A = torch.nn.DataParallel(model_A, device_ids=range(args.gpus)).cuda()
    model_FTFA = torch.nn.DataParallel(model_FTFA, device_ids=range(args.gpus)).cuda()
    model_erf = torch.nn.DataParallel(model_erf, device_ids=range(args.gpus)).cuda()
    #model = models.ERFNet(5)
    #model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()
    '''
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    '''
    loadA = 'trained/54/model_A.pth.tar'
    if os.path.isfile(loadA):
        print(("=> loading checkpoint '{}'".format(loadA)))
        load_filename = loadA
        checkpoint = torch.load(load_filename)
        
        model_A.load_state_dict(checkpoint)
    
    loadFA =  'trained/54/model_FTFA.pth.tar'
    if os.path.isfile(loadFA):
        print(("=> loading checkpoint '{}'".format(loadFA)))
        load_filename = loadFA
        checkpoint = torch.load(load_filename)
        
        model_FTFA.load_state_dict(checkpoint)         
    
    load2 = 'trained/54/model_erf.pth.tar'
    if os.path.isfile(load2):
        print(("=> loading checkpoint '{}'".format(load2)))
        load_filename = load2
        checkpoint = torch.load(load_filename) 
        
        model_erf.load_state_dict(checkpoint) 
    

    
    class model(nn.Module):    
        def __init__(self, num_classes):  # use encoder to pass pretrained encoder
            super().__init__()
            
            #self.model_A = model_A
            #self.model_FTFA = model_FTFA
            #self.model_erf = model_erf
            self.model_A = models.ResG(3,3,'A')
            self.model_A = torch.nn.DataParallel(self.model_A, device_ids=range(args.gpus)).cuda()
            self.model_FTFA = models.FTF(3,3)
            self.model_FTFA = torch.nn.DataParallel(self.model_FTFA, device_ids=range(args.gpus)).cuda()
            self.model_erf = models.ERFNet3(5)
            self.model_erf = torch.nn.DataParallel(self.model_erf, device_ids=range(args.gpus)).cuda()
            
            checkpointA = torch.load(loadA)
            self.model_A.load_state_dict(checkpointA)
            checkpointFA = torch.load(loadFA)
            self.model_FTFA.load_state_dict(checkpointFA)
            checkpoint2 = torch.load(load2)
            self.model_erf.load_state_dict(checkpoint2)
            
    
            
        def forward(self, input, only_encode=False):
            output1, output2, output3, output4, output5, output6 = self.model_A(input)
            output_up, output_down = self.model_FTFA(output1, output2, output3, output4, output5, output6)            
            return self.model_erf(output_up)    
    
    model = model(5)
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()
    
    cudnn.benchmark = True
    cudnn.fastest = True

    if args.mode == 0:  # mode 0 for video
        cap = cv2.VideoCapture(cap_name)
        while(True):
            check, in_frame_src = cap.read()
            if check:
                test(model, in_frame_src)
            else:
                print("Last frame")
                break

    elif args.mode == 1: # mode 1 for test image
        #with open(file_path, "r") as file:
        #for idx, line in enumerate(file):
        #if (idx + 1)% 3 == 1:          
        #line = line.strip()        
        #image_src = cv2.imread('/home/user/Datasets/culane'+line)
        #print('/home/user/Datasets/culane'+line)
        #test(model, image_src)
        #test(model_A, model_FTFA, model_erf, image_src, input_mean, input_std, line)
        #cv2.waitKey(0)
        image_src = cv2.imread(image)
        #test(model, image_src)
        test(model, image_src, input_mean, input_std)
        #cv2.waitKey(0)

#def test(model, image_src):
#def test(model_A, model_FTFA, model_erf, image_src, input_mean, input_std):
def test(model, image_src, input_mean, input_std):

    image_path = '/home/user/Datasets/culane/driver_100_30frame/05250343_0278.MP4/05220.jpg'    

    image_or = cv2.imread(image_path, 1)[:, :, ::-1]
    image_or = image_or[240: , :] 
    image_or = cv2.resize(image_or, (976,208), interpolation=cv2.INTER_LINEAR)
    rgb_img = np.float32(image_or) / 255 


    in_frame_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    # Input
    in_frame = cv2.resize(in_frame_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    croppedImage = in_frame[VERTICAL_CROP_SIZE:, :, :]  # FIX IT
    croppedImageTrain = cv2.resize(croppedImage, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)

    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        #tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
    ]
    )

    image = input_transform(croppedImageTrain)
    image = image.unsqueeze(0)
    input_var = torch.autograd.Variable(image)
    #print(model)

    # Comput  
    output = model(input_var)     #模型輸出
    output = F.softmax(output, dim=1)

    sem_classes = [
   'background', 'lane1', 'lane2', 'lane3', 'lane4' 
    ]
    
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    lane_category = sem_class_to_idx["lane4"]                #選擇要的類別

    lane1_mask = output[0, :, :, :].argmax(axis=0).detach().cpu().numpy()

    #lane1_mask_uint8 = 255 * np.uint8(lane1_mask == lane_category)
    lane1_mask_float = np.float32(lane1_mask == lane_category)   
    #txt_filename = 'output2.txt'
    #np.savetxt(txt_filename, lane1_mask_float, fmt='%d')


    #both_images = np.hstack((image_or, np.repeat(lane1_mask_uint8[:, :, None], 3, axis=-1)))
    #b_image = Image.fromarray(both_images)
    #b_image.save('demo/test4.jpg')
    
    from pytorch_grad_cam import GradCAM
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
        
        def __call__(self, model_output):
            return (model_output[self.category, :, : ] * self.mask).sum()

    

    target_layers = [model.module.model_FTFA.module.layer1]     #選擇特定的網路層
    targets = [SemanticSegmentationTarget(lane_category, lane1_mask_float)]

    with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    
        grayscale_cam = cam(input_tensor=input_var,             #計算grad cam
                        targets=targets)[0, :]
                        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        c_image = Image.fromarray(cam_image)
        c_image.save('demo/test_lane4.jpg')
    
    
    
    
    
    
    
    '''
    for l in range(LANES_COUNT):
        prob_map = (pred[0][l + 1] * 255).astype(int)
        prob_map = cv2.blur(prob_map, (9, 9))
        prob_map = prob_map.astype(np.uint8)
        maps.append(prob_map)
        mapsResized.append(cv2.resize(prob_map, (IN_IMAGE_W, IN_IMAGE_H_AFTER_CROP), interpolation=cv2.INTER_LINEAR))
        img = ptl.AddMask(img, prob_map, COLORS[l])  # Image with probability map

        exists.append(pred_exist[0][l] > 0.5)
        lines = ptl.GetLines(exists, maps)

    print(exists)
    res_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    #cv2.imshow("result_pb", res_img)

    for l in range(LANES_COUNT):
        points = lines[l]  # Points for the lane
        for point in points:
            cv2.circle(image_src, point, 5, POINT_COLORS[l], -1)

    #cv2.imshow("result_points", image_src)
    cv2.waitKey(100)
    
    directory = 'demo' 

    if not os.path.exists(directory):
                os.makedirs(directory)
    file_path = line            
    middle_path = file_path.split(".MP4")[0] + ".MP4" 
    full_file_path = 'demo/54' + middle_path
    #full_file_path = os.path.join(file_path.strip("/"))
    print(middle_path)
    #folder_path = os.path.dirname(full_file_path)
    if not os.path.exists(full_file_path):
        os.makedirs(full_file_path)            
    cv2.imwrite('demo/54/'+file_path, res_img)            
                
    #cv2.imwrite('demo/54/01890.jpg', res_img)
    '''


if __name__ == '__main__':
    main()
