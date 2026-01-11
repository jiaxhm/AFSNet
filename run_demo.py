# coding=gbk
import argparse
import os
import torch.backends.cudnn as cudnn
import models
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
from loss import *
import time
import random
from glob import glob

import matplotlib.pyplot as plt

# import sys
# sys.path.append('../cython')
# from connectivity import enforce_connectivity


'''
Infer from custom dataset:
author:Fengting Yang 
last modification: Mar.5th 2020

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output

'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='D:\\NEW\\awesome-semantic-segmentation-pytorch-master\core\data\datasets\\voc\VOC2012\JPEGImages2', help='path to images folder')
parser.add_argument('--data_suffix',  default='jpg', help='suffix of the testing image')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                                    default= 'F:\hysilu_adam_500000epochs_epochSize6000_b16_lr5e-05_posW0.003_23_12_21_09_40/model_best.tar')
parser.add_argument('--output', metavar='DIR', default= 'D:\\NEW\\awesome-semantic-segmentation-pytorch-master\core\data\datasets\\voc\VOC2012\JPEGImages3' , help='path to output folder')

parser.add_argument('--downsize', default=16, type=float,help='superpixel grid cell, must be same as training setting')

parser.add_argument('-nw', '--num_threads', default=16, type=int,  help='num_threads')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')

args = parser.parse_args()

#random.seed(100)#所设置的种子100范围内调用random()模块生成随机数
@torch.no_grad()
def test(args, model, img_paths, save_path, idx):
      # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])#标准化处理之后，可以使数据更好的响应激活函数，提高数据的表现力，减少梯度爆炸和梯度消失的出现。

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imread(load_path)[:, :, :3]#H*W*RGB
    # img_ = cv2.resize(img_, (int(img_.shape[1] * 0.5), int(img_.shape[0] * 0.5)), interpolation=cv2.INTER_CUBIC)
    # img_bgr = cv2.imread(load_path, 1)
    # img_ = img_bgr[:, :, [2, 1, 0]]
    # H, W, _ = img_.shape
    # H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)#np.ceil向上取整
    #
    # # get spixel id
    # n_spixl_h = int(np.floor(H_ / args.downsize))
    # n_spixl_w = int(np.floor(W_ / args.downsize))
    #
    # spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    n_spixl_h = int(np.floor(160 / args.downsize))
    n_spixl_w = int(np.floor(320 / args.downsize))
    spix_values = np.int32(np.arange(0, n_spixl_w*n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)#9*(H/16)*（W/16）

    spix_idx_tensor = np.repeat(
    np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)#先在维度1重复16次，再在维度2重复16次   9*H*W
    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()#1*9*H*W
    n_spixel =  int(n_spixl_h * n_spixl_w)#超像素个数


    img = cv2.resize(img_, (160,320), interpolation=cv2.INTER_CUBIC)#H*W*3
    img1 = input_transform(img)#img1是经过三次插值转换的张量
    ori_img = input_transform(img_)#ori_img是原图张量


    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))#升维,1*3*H*W
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( 160,320), mode='nearest').type(torch.int)


    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)#spixel_viz是分割好的图像数组，spixel_label_map是标签数组

    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))


    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '.jpg')
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))


    # # save spixel label map
    # if not os.path.isdir(os.path.join(save_path, 'spixel_label_map')):
    #     os.makedirs(os.path.join(save_path, 'spixel_label_map'))
    # spixllabel_save_name = os.path.join(save_path, 'spixel_label_map', imgId + '_sPixel.png')
    # imsave(spixllabel_save_name, spixel_label_map.transpose(1, 2, 0))
########

#特征图

    # save the unique maps as csv, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'map_csv')):
    #     os.makedirs(os.path.join(save_path, 'map_csv'))
    # output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    #   # plus 1 to make it consistent with the toolkit format
    # np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i',delimiter=",")


    if idx % 10 == 0:
        print("processing %d"%idx)

    return toc



def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    save_path = args.output
    print('=> will save everything to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tst_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    tst_lst.sort()#排序

    if len(tst_lst) == 0:
        print('Wrong data dir or suffix!')
        exit(1)

    print('{} samples found'.format(len(tst_lst)))

    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True



    mean_time = 0
    for n in range(len(tst_lst)):
      time = test(args, model, tst_lst, save_path, n)
      mean_time += time
    print("avg_time per img: %.3f"%(mean_time/len(tst_lst)))

if __name__ == '__main__':
    main()
