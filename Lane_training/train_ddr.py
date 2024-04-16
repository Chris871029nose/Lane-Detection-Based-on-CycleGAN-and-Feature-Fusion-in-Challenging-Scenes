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
import dataset as ds
from options.options import parser
import itertools

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
        ignore_label = 255  # 0
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37  # merge the noise and ignore labels
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
    #model = models.FTFERFNet(num_class)  
    model_ddr = models.DualResNet(models.BasicBlock, [3, 4, 6, 3], num_classes=5, planes=64, spp_planes=128, head_planes=256, augment=False)
    input_mean = model_ddr.input_mean
    input_std = model_ddr.input_std
    model_A = torch.nn.DataParallel(model_A, device_ids=range(args.gpus)).cuda()
    model_B = torch.nn.DataParallel(model_B, device_ids=range(args.gpus)).cuda()
    model_FTFA = torch.nn.DataParallel(model_FTFA, device_ids=range(args.gpus)).cuda()
    model_FTFB = torch.nn.DataParallel(model_FTFB, device_ids=range(args.gpus)).cuda()
    model_ddr = torch.nn.DataParallel(model_ddr, device_ids=range(args.gpus)).cuda()
    


    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        ckpt_name = []
        cnt = 0
        for name, param in state_dict.items():
            if name not in list(own_state.keys()) or 'output_conv' in name:
                 ckpt_name.append(name)
                 continue
            own_state[name].copy_(param)
            cnt += 1
        print('#reused param: {}'.format(cnt))
        return model

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model = load_my_state_dict(model, checkpoint['state_dict'])
            # torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.train_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            #tf.GroupRandomCropRatio(size=(976, 208)),
            tf.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (ignore_label, ))),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
        
    train_loader2 = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet2')(data_list=args.train_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            #tf.GroupRandomCropRatio(size=(976, 208)),
            tf.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (ignore_label, ))),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_con = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.SGD(itertools.chain(model_FTFA.parameters(), model_FTFB.parameters(), model_ddr.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(itertools.chain(model_erf.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer1 = torch.optim.SGD(itertools.chain(model_A.parameters(), model_erf.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.SGD(itertools.chain(model_B.parameters(), model_erf.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model_erf.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    args.evaluate = False

    if args.evaluate:
        validate(val_loader, model_A, model_B, model_ddr, criterion, 0, evaluator)
        #save_checkpoint2(is_best)
        return
        
    load_FTFA = 'trained/109/model_FTFA.pth.tar'
    if os.path.isfile(load_FTFA):
        print(("=> loading checkpoint '{}'".format(load_FTFA)))
        load_filename = load_FTFA
        checkpoint = torch.load(load_filename)        
        model_FTFA.load_state_dict(checkpoint)  
        
    load_FTFB = 'trained/109/model_FTFB.pth.tar'
    if os.path.isfile(load_FTFB):
        print(("=> loading checkpoint '{}'".format(load_FTFB)))
        load_filename = load_FTFB
        checkpoint = torch.load(load_filename)        
        model_FTFB.load_state_dict(checkpoint)          
        
    load_ddr = 'trained/109/model_ddr.pth.tar'
    if os.path.isfile(load_ddr):
        print(("=> loading checkpoint '{}'".format(load_ddr)))
        load_filename = load_ddr
        checkpoint = torch.load(load_filename)        
        model_ddr.load_state_dict(checkpoint)            
       

    for epoch in range(args.epochs):  # args.start_epoch
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, train_loader2, model_A, model_B, model_FTFA, model_FTFB, model_ddr, criterion, criterion_exist, criterion_con, optimizer, epoch)
        is_best = 1
        save_checkpoint2(model_A.state_dict(), model_B.state_dict(), model_FTFA.state_dict(), model_FTFB.state_dict(), model_ddr.state_dict(), is_best)
        
        '''
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            mIoU = validate(val_loader,  model_A, model_B, model_erf, criterion, (epoch + 1) * len(train_loader), evaluator)
            # remember best mIoU and save checkpoint
            is_best = mIoU > best_mIoU
            best_mIoU = max(mIoU, best_mIoU)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict1': model_A.state_dict(),
                'state_dict2': model_B.state_dict(),
                'state_dict3': model_erf.state_dict(),
                'best_mIoU': best_mIoU,
            }, is_best)
            '''
            #save_checkpoint2(model_A.state_dict(), model_B.state_dict(), model_erf.state_dict(), is_best)
            


def train(train_loader, train_loader2, model_A, model_B, model_FTFA, model_FTFB, model_ddr, criterion, criterion_exist, criterion_con, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_up = AverageMeter()
    losses_exist_up = AverageMeter()
    losses_down = AverageMeter()
    losses_exist_down = AverageMeter()
    losses_con = AverageMeter()

    # switch to train mode
    model_A.eval()
    model_B.eval()
    model_FTFA.train()
    model_FTFB.train()
    model_ddr.train()
    end = time.time()
      
    for i, (input, target, target_exist) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        target = target.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input)
        #input_var = torch.tensor(input)
        target_var = torch.autograd.Variable(target)
        target_exist_var = torch.autograd.Variable(target_exist)
        
        with torch.no_grad():
            output1, output2, output3, output4, output5, output6 = model_A(input_var)
        output_up, output_down = model_FTFA(output1, output2, output3, output4, output5, output6)
        output_up, output_exist_up = model_ddr(output_up)  # output_mid
        output_down, output_exist_down = model_ddr(output_down)  # output_mid
        
        loss_up = criterion(torch.nn.functional.log_softmax(output_up, dim=1), target_var)
        loss_down = criterion(torch.nn.functional.log_softmax(output_down, dim=1), target_var)
        loss_exist_up = criterion_exist(output_exist_up, target_exist_var)
        loss_exist_down = criterion_exist(output_exist_down, target_exist_var)
        
        loss_con = criterion_con(torch.nn.functional.log_softmax(output_up, dim=1), torch.nn.functional.log_softmax(output_down, dim=1))
        #loss_con = 0
        loss_tot = loss_up + loss_down + loss_exist_up * 0.1 + loss_exist_down * 0.1 + loss_con
                
        # measure accuracy and record loss
        losses_up.update(loss_up.data.item(), input.size(0))
        losses_exist_up.update(loss_exist_up.item(), input.size(0))
        losses_down.update(loss_down.data.item(), input.size(0))
        losses_exist_down.update(loss_exist_down.item(), input.size(0))
        losses_con.update(loss_con.data.item(), input.size(0))
        #losses_con = 0.0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print((
                    'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss_up {loss_up.val:.4f} ({loss_up.avg:.4f})\t' 'Loss_down {loss_down.val:.4f} ({loss_down.avg:.4f})\t' 'Loss_con {loss_con.val:.4f} ({loss_con.avg:.4f})\t' 'Loss_exist_up {loss_exist_up.val:.4f} ({loss_exist_up.avg:.4f})\t' 'Loss_exist_down {loss_exist_down.val:.4f} ({loss_exist_down.avg:.4f})\t'.format(
                         epoch+1, i+1, len(train_loader2), batch_time=batch_time, data_time=data_time, loss_up=losses_up, loss_down=losses_down, loss_con=losses_con,
                         loss_exist_up=losses_exist_up, loss_exist_down=losses_exist_down, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses_up.reset()
            losses_down.reset()
            losses_con.reset()
        if (i + 1) == 10:    
            is_best = 1            
            save_checkpoint2(model_A.state_dict(), model_B.state_dict(), model_FTFA.state_dict(), model_FTFB.state_dict(), model_ddr.state_dict(), is_best)
            print('save=10' )
        if (i + 1) == 50:    
            is_best = 1        
            save_checkpoint2(model_A.state_dict(), model_B.state_dict(), model_FTFA.state_dict(), model_FTFB.state_dict(), model_ddr.state_dict(), is_best)                
            print('save=50' )
            
    for i, (input, target, target_exist) in enumerate(train_loader2):
        data_time.update(time.time() - end)
        
        target = target.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input)
        #input_var = torch.tensor(input)
        target_var = torch.autograd.Variable(target)
        target_exist_var = torch.autograd.Variable(target_exist)
        
        with torch.no_grad():
            output1, output2, output3, output4, output5, output6 = model_B(input_var)
        output_up, output_down = model_FTFB(output1, output2, output3, output4, output5, output6)
        output_up, output_exist_up = model_ddr(output_up)  # output_mid
        output_down, output_exist_down = model_ddr(output_down)  # output_mid
        
        loss_up = criterion(torch.nn.functional.log_softmax(output_up, dim=1), target_var)
        loss_down = criterion(torch.nn.functional.log_softmax(output_down, dim=1), target_var)
        loss_exist_up = criterion_exist(output_exist_up, target_exist_var)
        loss_exist_down = criterion_exist(output_exist_down, target_exist_var)
        
        loss_con = criterion_con(torch.nn.functional.log_softmax(output_up, dim=1), torch.nn.functional.log_softmax(output_down, dim=1))
        #loss_con = 0
        loss_tot = loss_up + loss_down + loss_exist_up * 0.1 + loss_exist_down * 0.1  + loss_con
                
        losses_up.update(loss_up.data.item(), input.size(0))
        losses_exist_up.update(loss_exist_up.item(), input.size(0))
        losses_down.update(loss_down.data.item(), input.size(0))
        losses_exist_down.update(loss_exist_down.item(), input.size(0))
        losses_con.update(loss_con.data.item(), input.size(0))
        #losses_con = 0.0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()
    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print((
                    'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss_up {loss_up.val:.4f} ({loss_up.avg:.4f})\t' 'Loss_down {loss_down.val:.4f} ({loss_down.avg:.4f})\t' 'Loss_con {loss_con.val:.4f} ({loss_con.avg:.4f})\t' 'Loss_exist_up {loss_exist_up.val:.4f} ({loss_exist_up.avg:.4f})\t' 'Loss_exist_down {loss_exist_down.val:.4f} ({loss_exist_down.avg:.4f})\t'.format(
                         epoch+1, i+1, len(train_loader2), batch_time=batch_time, data_time=data_time, loss_up=losses_up, loss_down=losses_down, loss_con=losses_con,
                         loss_exist_up=losses_exist_up, loss_exist_down=losses_exist_down, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses_up.reset()
            losses_down.reset()
            losses_con.reset()
       
    '''  
    for i, (input, target, target_exist) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_exist_var = torch.autograd.Variable(target_exist)

        # compute output
        if i <= 1723:
            output = model_A(input_var)
            output, output_exist = model_erf(output)  # output_mid
        else:
            output = model_B(input_var)
            output, output_exist = model_erf(output)  # output_mid
            
        loss = criterion(torch.nn.functional.log_softmax(output, dim=1), target_var)
        loss_exist = criterion_exist(output_exist, target_exist_var)
        loss_tot = loss + loss_exist * 0.1
    
        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        losses_exist.update(loss_exist.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print((
                      'Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'.format(
                          epoch, i, 14814, batch_time=batch_time, data_time=data_time, loss=losses,
                          loss_exist=losses_exist, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses.reset()
       '''


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(val_loader,  model_A, model_B, model_erf, criterion, iter, evaluator, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()

    # switch to evaluate mode
    #model.eval()

    end = time.time()
    for i, (input, target, target_exist) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        #output, _ = model(input_var)
        output = model_B(input_var)
        output, output_exist = model_erf(output)
        loss = criterion(torch.nn.functional.log_softmax(output, dim=1), target_var)

        # measure accuracy and record loss

        pred = output.data.cpu().numpy().transpose(0, 2, 3, 1)
        pred = np.argmax(pred, axis=3).astype(np.uint8)
        IoU.update(evaluator(pred, target.cpu().numpy()))
        losses.update(loss.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
            mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
            mIoU = np.sum(mIoU) / len(mIoU)
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Pixels Acc {acc:.3f}\t' 'mIoU {mIoU:.3f}'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc, mIoU=mIoU)))

    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
    mIoU = np.sum(mIoU) / len(mIoU)
    print(('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})\tLoss {loss.avg:.5f}'.format(acc=acc, mIoU=mIoU, bestmIoU=max(mIoU, best_mIoU), loss=losses)))

    return mIoU


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('trained'):
        os.makedirs('trained')
    filename = os.path.join('trained', '_'.join((args.snapshot_pref, args.method.lower(), filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join('trained', '_'.join((args.snapshot_pref, args.method.lower(), 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)
        
def save_checkpoint2(state_A, state_B, state_FTFA, state_FTFB, state_ddr, is_best):
    if not os.path.exists('trained'):
        os.makedirs('trained')
    #filename = os.path.join('trained', '_'.join((args.snapshot_pref, args.method.lower(), filename)))
    filename_A =  os.path.join('trained', 'model_A.pth.tar')
    filename_B =  os.path.join('trained', 'model_B.pth.tar')
    filename_FTFA = os.path.join('trained', 'model_FTFA.pth.tar')
    filename_FTFB = os.path.join('trained', 'model_FTFB.pth.tar')
    filename_ddr =  os.path.join('trained', 'model_ddr.pth.tar')
    #torch.save(state, filename)
    torch.save(state_A, filename_A)
    torch.save(state_B, filename_B)
    torch.save(state_FTFA, filename_FTFA)
    torch.save(state_FTFB, filename_FTFB)
    torch.save(state_ddr, filename_ddr)
    
    '''
    if is_best:
        #best_name = os.path.join('trained', '_'.join((args.snapshot_pref, args.method.lower(), 'model_best.pth.tar')))
        #shutil.copyfile(filename, best_name)
        best_name_A = os.path.join('trained', 'model_best_A.pth.tar')
        best_name_B = os.path.join('trained', 'model_best_B.pth.tar')
        best_name_FTFA = os.path.join('trained', 'model_best_FTFA.pth.tar')
        best_name_FTFB = os.path.join('trained', 'model_best_FTFB.pth.tar')
        
        best_name_erf = os.path.join('trained', 'model_best_erf.pth.tar')
        shutil.copyfile(filename_A, best_name_A)
        shutil.copyfile(filename_B, best_name_B)
        shutil.copyfile(filename_erf, best_name_erf)
    '''


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
        hs = np.bincount(sumim[locs], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay


if __name__ == '__main__':
    main()
