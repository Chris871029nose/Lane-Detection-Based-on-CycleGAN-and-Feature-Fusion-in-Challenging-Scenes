import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numbers

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        self.data_list = 'train_gt'
        self.phase_list = 'list'
        self.dir_A_list = os.path.join(self.dir_A, self.phase_list)
        self.dir_B_list = os.path.join(self.dir_B, self.phase_list)
        
        with open(os.path.join(self.dir_A_list, self.data_list + '.txt')) as f:
            self.A_paths = []
            #self.A_paths_label = []
            #self.A_paths_exist = []
            for line in f:
                self.A_paths.append(self.dir_A + line.strip().split(" ")[0])
                #self.A_paths_label.append(self.dir_A.replace('/trainA','/labelA') + line.strip().split(" ")[1])
                #self.A_paths_exist.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))
                
        with open(os.path.join(self.dir_B_list, self.data_list + '.txt')) as g:
            self.B_paths = []
            #self.B_paths_label = []
            #self.B_paths_exist = []
            for line in g:
                self.B_paths.append(self.dir_B + line.strip().split(" ")[0])
                #self.B_paths_label.append(self.dir_B.replace('/trainB','/labelB') + line.strip().split(" ")[1])
                #self.B_paths_exist.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))
                
        #self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        #self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image   
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        
        #self.transform_L = get_transform2(self.opt, grayscale=(output_nc == 1))
        #self.transform_L = get_transform2(self.opt, grayscale= True)
        '''
        self.phase_label = 'label'
        self.dir_A_label = os.path.join(opt.dataroot, self.phase_label+'A')
        self.A_paths_label = sorted(make_dataset(self.dir_A_label, opt.max_dataset_size))
        #
        self.dir_B_label = os.path.join(opt.dataroot, self.phase_label+'B')
        self.B_paths_label = sorted(make_dataset(self.dir_B_label, opt.max_dataset_size))
        #
        self.data_list = 'train_gt'
        self.phase_list = 'list'
        self.dir_A_list = os.path.join(self.dir_A, self.phase_list)
        with open(os.path.join(self.dir_A_list, self.data_list + '.txt')) as f:
            self.exist_list_A = []
            for line in f:
                self.exist_list_A.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))
        #
        self.dir_B_list = os.path.join(self.dir_B, self.phase_list)
        with open(os.path.join(self.dir_B_list, self.data_list + '.txt')) as g:
            self.exist_list_B = []
            for line in g:
                self.exist_list_B.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))        
        #
        '''

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        
        
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        '''
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        '''
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        #A_img = A_img.crop((2, 7, 818, 295))
        #B_img = B_img.crop((2, 7, 818, 295))
        #A_img = A_img.crop((0, 240, 1640, 590 ))
        #B_img = B_img.crop((0, 240, 1640, 590))
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        '''                           
        #add label & lane exist
        A_path_label = self.A_paths_label[index % self.A_size]

        
        B_path_label = self.B_paths_label[index % self.B_size]
        
        A_img2 = cv2.imread(A_path).astype(np.float32)
        B_img2 = cv2.imread(B_path).astype(np.float32)

        A_img2 = A_img2[240:,:, :] 
        B_img2 = B_img2[240:,:, :] 
        A_img2 = cv2.resize(A_img2, (976, 208), interpolation=cv2.INTER_NEAREST)
        B_img2 = cv2.resize(B_img2, (976, 208), interpolation=cv2.INTER_NEAREST)
        

        A_label = cv2.imread(A_path_label, cv2.IMREAD_UNCHANGED)
        B_label = cv2.imread(B_path_label, cv2.IMREAD_UNCHANGED)

        A_label = A_label[240:,:]
        B_label = B_label[240:,:]
        A_label = cv2.resize(A_label, (976, 208), interpolation=cv2.INTER_NEAREST)
        B_label = cv2.resize(B_label, (976, 208), interpolation=cv2.INTER_NEAREST)
        A_label = A_label.squeeze()
        B_label = B_label.squeeze()

        ignore_label = 255
        input_mean = [0.5, 0.5, 0.5]  # [0, 0, 0]
        input_std =  [0.5, 0.5, 0.5]
        
        
        self.transform = transforms.Compose([
            GroupRandomCropRatio(size=(976, 208)),
            GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])

        A2, A_label = self.transform((A_img2, A_label))
        B2, B_label = self.transform((B_img2, B_label))

        A_label = torch.from_numpy(A_label).contiguous().long()
        B_label = torch.from_numpy(B_label).contiguous().long()
        
        exist_path_A = self.A_paths_exist[index % self.A_size]
        A_exist = exist_path_A
        exist_path_B = self.B_paths_exist[index % self.B_size]
        B_exist = exist_path_B
        '''                        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
                                       
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)   
        
      

class GroupRandomCropRatio(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        #h, w = 576, 1632
        tw, th = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
            #out_images.append(img[h1:h2])
        return out_images     
        

            
class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_group):
        out_images = list()
        for img, m, s in zip(img_group, self.mean, self.std):
            #img = img - np.array(m)  # single channel image
            #img = img / np.array(s)
            
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            
            out_images.append(img)

        return out_images
