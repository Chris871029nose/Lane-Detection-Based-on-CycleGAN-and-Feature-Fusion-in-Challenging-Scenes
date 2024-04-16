import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
#from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 
        ignore_label = 255 
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model_A = models.ResG(3,3,'A')
    model_B = models.ResG(3,3,'B')
    model_FTFA = models.FTF(3,3)
    model_FTFB = models.FTF(3,3)
    #model_erf = models.ERFNet3(num_class)
    model_ddr = models.DualResNet(models.BasicBlock, [3, 4, 6, 3], num_classes=5, planes=64, spp_planes=128, head_planes=256, augment=False)
    input_mean = model_ddr.input_mean
    input_std = model_ddr.input_std
    #policies = model_erf.get_optim_policies()
    model_A = torch.nn.DataParallel(model_A, device_ids=range(args.gpus)).cuda()
    model_B = torch.nn.DataParallel(model_B, device_ids=range(args.gpus)).cuda()
    model_FTFA = torch.nn.DataParallel(model_FTFA, device_ids=range(args.gpus)).cuda()
    model_FTFB = torch.nn.DataParallel(model_FTFB, device_ids=range(args.gpus)).cuda()
    #model_erf = torch.nn.DataParallel(model_erf, device_ids=range(args.gpus)).cuda()
    model_ddr = torch.nn.DataParallel(model_ddr, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            load_filename = args.resume
            checkpoint = torch.load(load_filename)
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            re_model = OrderedDict()
            #re_en = OrderedDict()
            #re_de = OrderedDict()
            '''
            for k, v in checkpoint.items():
                name = k.replace('module.model','model')
                re_model[name] = v
            for k, v in re_model.items():
                name = k.replace('module.layer','layer')
                new_state_dict[name] =v
            model.load_state_dict(new_state_dict)
            '''
            model_A.load_state_dict(checkpoint)
            
            
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            
    loadB = 'trained/model_B.pth.tar'
    if os.path.isfile(loadB):
        print(("=> loading checkpoint '{}'".format(loadB)))
        load_filename = loadB
        checkpoint = torch.load(load_filename)
        
        model_B.load_state_dict(checkpoint)
        
    loadFA =  'trained/model_FTFA.pth.tar'
    if os.path.isfile(loadFA):
        print(("=> loading checkpoint '{}'".format(loadFA)))
        load_filename = loadFA
        checkpoint = torch.load(load_filename)
        
        model_FTFA.load_state_dict(checkpoint)     
        
    loadFB =  'trained/model_FTFB.pth.tar'
    if os.path.isfile(loadFB):
        print(("=> loading checkpoint '{}'".format(loadFB)))
        load_filename = loadFB
        checkpoint = torch.load(load_filename)
        
        model_FTFB.load_state_dict(checkpoint)            
            
    load2 = 'trained/model_ddr.pth.tar'
    if os.path.isfile(load2):
        print(("=> loading checkpoint '{}'".format(load2)))
        load_filename = load2
        checkpoint = torch.load(load_filename)
            
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        #re_FTF = OrderedDict()
        re_en = OrderedDict()
        re_de = OrderedDict()
        '''
        for k, v in checkpoint.items():
            name = k.replace('FTF','module.FTF')
            new_state_dict[name] = v
        '''    
        '''
        for k, v in checkpoint.items():
            name = k.replace('encoder.encoder','module.encoder.encoder')
            re_en[name] = v
        for k, v in re_en.items():
            name = k.replace('decoder','module.decoder')
            re_de[name] = v
        for k, v in re_de.items():
            name = k.replace('lane','module.lane')
            new_state_dict[name] = v
                
        model_erf.load_state_dict(new_state_dict)   
        '''
        model_ddr.load_state_dict(checkpoint) 
         


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    #for group in policies:
    #    print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    #optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model_A, model_B, model_FTFA, model_FTFB, model_ddr, criterion, 0, evaluator)
    return


def validate(val_loader, model_A, model_B,  model_FTFA, model_FTFB, model_ddr, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model_A.eval()
    model_B.eval()
    model_FTFA.eval()
    model_FTFB.eval()
    model_ddr.eval()

    end = time.time()
    for i, (input, target, img_name) in enumerate(val_loader):

        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        #output_GAN, output, output_exist = model(input_var)
        #output, output_exist = model(input_var)
        #if i <= 702:
        #    output_up, output_down = model_A(input_var)
        #else:
        #    output_up, output_down = model_B(input_var)
        with torch.no_grad():
            if i <= 175:
                output1, output2, output3, output4, output5, output6 = model_A(input_var)
                output_up, output_down = model_FTFA(output1, output2, output3, output4, output5, output6)
            else:
                output1, output2, output3, output4, output5, output6 = model_B(input_var)
                output_up, output_down = model_FTFB(output1, output2, output3, output4, output5, output6)
                
            output_up, output_exist_up = model_ddr(output_up)  # output_mid
            output_down, output_exist_down = model_ddr(output_down)  # output_mid
            
        #output, output_exist = model_erf(output)
        #output_up, output_exist_up = model_erf(output_up)  # output_mid
        #output_down, output_exist_down = model_erf(output_down)  # output_mid
        
        # measure accuracy and record loss

        #output = F.softmax(output, dim=1)
        output_up = F.softmax(output_up, dim=1)
        output_down = F.softmax(output_down, dim=1)

        #pred = output.data.cpu().numpy() # BxCxHxW
        #pred_exist = output_exist.data.cpu().numpy() # BxO
        pred = output_up.data.cpu().numpy() # BxCxHxW
        pred_exist = output_exist_up.data.cpu().numpy() # BxO
        pred_down = output_down.data.cpu().numpy() # BxCxHxW
        pred_exist_down = output_exist_down.data.cpu().numpy() # BxO

        for cnt in range(len(img_name)):
            directory = 'predicts/ERFNet' + img_name[cnt][:-10]
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open('predicts/ERFNet'+img_name[cnt].replace('.jpg', '.exist.txt'), 'w')
            for num in range(4):
                prob_map = (pred[cnt][num+1]*255).astype(int)
                save_img = cv2.blur(prob_map,(9,9))
                cv2.imwrite('predicts/ERFNet'+img_name[cnt].replace('.jpg', '_'+str(num+1)+'_avg.png'), save_img)
                if pred_exist[cnt][num] > 0.5:
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
            file_exist.close()
            
        for cnt in range(len(img_name)):
            directory = 'predicts2/ERFNet' + img_name[cnt][:-10]
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open('predicts2/ERFNet'+img_name[cnt].replace('.jpg', '.exist.txt'), 'w')
            for num in range(4):
                prob_map = (pred_down[cnt][num+1]*255).astype(int)
                save_img = cv2.blur(prob_map,(9,9))
                cv2.imwrite('predicts2/ERFNet'+img_name[cnt].replace('.jpg', '_'+str(num+1)+'_avg.png'), save_img)
                if pred_exist_down[cnt][num] > 0.5:
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
            file_exist.close()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
