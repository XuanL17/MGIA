# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from gensim.models import word2vec
import pandas as pd
import re
from utils import label_to_onehot, cross_entropy_for_onehot

from utils import label_to_onehot2, cross_entropy_for_onehot
from gensim.models import KeyedVectors
from sklearn.metrics import  accuracy_score
def gt_psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

def _get_meanstd(dataset):
    cc = torch.cat([dataset[i][0].reshape(3, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)


#--------------------------------------------------------------------------------------choose dataset
dst = datasets.CIFAR100(root='~/.torch', download=True)
dst2 = datasets.CIFAR100("~/.torch", download=True,transform=transforms.ToTensor())

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

dm,ds=_get_meanstd(dst2)
dm = torch.as_tensor(dm)[:, None, None].to(device)
ds = torch.as_tensor(ds)[:, None, None].to(device)
label_name_dict={19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak tree', 42: 'leopard', 47: 'maple tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweetpepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup truck', 16: 'can'}

model = word2vec.Word2Vec.load('../100features_text8')

true_label = []
image_pred_label = []
text_pred_label = []
fuse_pred_label = []
pred_label=[]
psnr_list=[]
image_loss = []

for i in range(50):
    # ------------------------------------------------Loading image data
    gt_data_image = tp(dst[i][0]).to(device)
    gt_data_image = gt_data_image.view(1, *gt_data_image.size())
    plt.imshow(tt(gt_data_image[0].cpu()))
    plt.show()

    # Shared label
    gt_label = torch.Tensor([dst[i][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label1 = label_to_onehot(gt_label)
    gt_onehot_label2 = label_to_onehot(gt_label)
    true_label.append(gt_label.item())

    # # --------------------------------------------------------------------Loading text data

    sentence = 'a ' + label_name_dict[gt_label.item()] + ' in the image'

    print('----------------------------original sentence:', sentence, ' label:', gt_label.item())
    sentence = str.lower(sentence)
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = sentence.strip().split()

    vec = np.array(model.wv[sentence])
    gt_data_text = torch.tensor(vec).unsqueeze(0).unsqueeze(1).to(device)

    # ----------------------------------Preparation
    from OURS.models.vision import LeNet, weights_init1
    net1 = LeNet().to(device)

    torch.manual_seed(1234)

    net1.apply(weights_init1)
    criterion1 = cross_entropy_for_onehot

    # # compute original gradient
    # pred1 = net1(gt_data_image)
    # y1 = criterion1(pred1, gt_onehot_label1)
    # dy_dx1 = torch.autograd.grad(y1, net1.parameters())
    #
    # original_dy_dx1 = list((_.detach().clone() for _ in dy_dx1))

    # generate dummy data and label
    dummy_data1 = torch.randn(gt_data_image.size()).to(device).requires_grad_(True)
    dummy_label1 = torch.randn(gt_onehot_label1.size()).to(device).requires_grad_(True)

    optimizer1 = torch.optim.LBFGS([dummy_data1, dummy_label1])

    from models.nlp import TextCNN, weights_init2

    # # # from models.resnet18 import Resnet18
    # #
    net2 = TextCNN().to(device)
    # print(net)
    # #
    torch.manual_seed(1234)
    # #
    # #
    # #
    criterion2 = cross_entropy_for_onehot
    # #
    # # # compute original gradient
    pred2 = net2(gt_data_text)

    y2 = criterion2(pred2, gt_onehot_label2)
    dy_dx2 = torch.autograd.grad(y2, net2.parameters())
    # #
    original_dy_dx2 = list((_.detach().clone() for _ in dy_dx2))
    #
    # # generate dummy data.txt and label
    dummy_data2 = torch.randn(gt_data_text.size()).to(device).requires_grad_(True)
    dummy_label2 = torch.randn(gt_onehot_label2.size()).to(device).requires_grad_(True)
    #
    #
    optimizer2 = torch.optim.LBFGS([dummy_data2, dummy_label2])
    #
    #
    # history = []
    for iters in range(100):
        def closure2():
            optimizer2.zero_grad()

            dummy_pred = net2(dummy_data2)
            dummy_onehot_label = F.softmax(dummy_label2, dim=-1)
            dummy_loss = criterion2(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net2.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx2):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer2.step(closure2)
        if iters % 10 == 0:
            recovery_ret = ''
            with torch.no_grad():
                recovered_wordvec = dummy_data2.squeeze(0).squeeze(0).cpu().numpy()
                for vec in recovered_wordvec:
                    temp = model.wv.most_similar(positive=[vec], topn=1)
                    recovery_ret += temp[0][0] + ' '
            current_loss = closure2()
            print(iters, "Text loss: %.4f" % current_loss.item(), 'recovered sentence: [', recovery_ret, ']')
    print(torch.argmax(dummy_label2).item())

    #
    print('--------------------------------------------------------------------------Image model')

    temp=3
    alpha=0.3

    # compute original gradient
    pred1 = net1(gt_data_image)
    orgiginal_student_loss = cross_entropy_for_onehot(pred1,gt_onehot_label1)
    original_distillation_loss = cross_entropy_for_onehot(pred1 / temp, gt_onehot_label2 / temp)
    origibal_dummy_loss = alpha * orgiginal_student_loss + (1 - alpha) * original_distillation_loss

    dy_dx1 = torch.autograd.grad(origibal_dummy_loss, net1.parameters())

    original_dy_dx1 = list((_.detach().clone() for _ in dy_dx1))
    history = []

    # Save the loss of the current image
    temp_loss = 0
    for iters in range(100):
        def closure1():
            optimizer1.zero_grad()
            dummy_pred = net1(dummy_data1)

            student_loss=cross_entropy_for_onehot(dummy_pred,F.softmax(dummy_label1,dim=-1))
            distillation_loss=cross_entropy_for_onehot(dummy_pred/temp,F.softmax(dummy_label2/temp,dim=-1))
            dummy_loss=alpha*student_loss+(1-alpha)*distillation_loss

            dummy_dy_dx = torch.autograd.grad(dummy_loss, net1.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx1):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer1.step(closure1)
        if iters % 10 == 0:
            current_loss1 = closure1()
            temp_loss=current_loss1

            print(iters, "Image loss: %.4f" % current_loss1.item())
            history.append(tt(dummy_data1[0].cpu()))

    image_loss.append(temp_loss.item())
    recoverd_number=[i for i in image_loss if i<0.015]

    dummy_label_image = dummy_label1.squeeze(0)
    dummy_label_text = dummy_label2.squeeze(0)


    image_pred_label.append(torch.argmax(dummy_label_image).item())
    text_pred_label.append(torch.argmax(dummy_label_text).item())

    dummy_label = (0.8* dummy_label_text + 0.2* dummy_label_image)

    fuse_pred_label.append(torch.argmax(dummy_label).item())
    print('image_pred_label', image_pred_label)
    print('text_pred_label', text_pred_label)
    print('fuse_pred_label', fuse_pred_label)
    print('true_label', true_label)

    print('Image label recovery accuracy:', accuracy_score(true_label, image_pred_label))
    print('Text label recovery accuracy:', accuracy_score(true_label, text_pred_label))
    print('Fuse label recovery accuracy:', accuracy_score(true_label, fuse_pred_label))
    print('image recu acc:',len(recoverd_number)/len(image_loss))
    with torch.no_grad():
        psnr = gt_psnr(gt_data_image, dummy_data1,factor=1/ds)
        if psnr > 0:
            psnr_list.append(psnr)
        else:
            psnr_list.append(0)
    print(psnr_list)

    plt.imshow(tt(dummy_data1[0].cpu()))
    plt.show()
print('avg psnr:', np.mean(psnr_list))




