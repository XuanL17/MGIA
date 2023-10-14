# -*- coding: utf-8 -*-
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import datasets, transforms
from gensim.models import word2vec
import re
from utils import label_to_onehot

from utils import cross_entropy_for_onehot
from sklearn.metrics import accuracy_score

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)
label_name_dict={19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak', 42: 'leopard', 47: 'maple', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm ', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweetpepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'truck', 16: 'can'}
model = word2vec.Word2Vec.load('../100features_text8')
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
# ---------------------------------------------------------------------------choose dataset
dst = datasets.CIFAR100("~/.torch", download=True)
dst2 = datasets.CIFAR100("~/.torch", download=True,transform=transforms.ToTensor())

tp = transforms.ToTensor()
tt = transforms.ToPILImage()
rs=transforms.Resize([32,32])


dm,ds=_get_meanstd(dst2)
dm = torch.as_tensor(dm)[:, None, None].to(device)
ds = torch.as_tensor(ds)[:, None, None].to(device)
psnr_list=[]

true_label=[]
pred_label=[]
image_loss = []
for i in range(50):

    gt_image_data = tp(rs(dst[i][0])).to(device)
    gt_image_data = gt_image_data.view(1, *gt_image_data.size())
    gt_label = torch.Tensor([dst[i][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    true_label.append(gt_label.item())

    # print(gt_data.shape)
    gt_onehot_label = label_to_onehot(gt_label)
    plt.imshow(tt(gt_image_data[0].cpu()))
    plt.show()

    # --------------------------------------------Loading text data
    # Load the word2vec pre-trained with the dataset
    # You can take a sentence from the dataset, or make a sentence yourself using words from the word list
    # # Read sentences and labels from a dataset
    # data = pd.read_csv('./sampled_eng_computercourse_review.csv')
    # sentence = data['comment'][0]
    # label = data['label'][0]

    # Construct your own sentences and labels
    sentence = 'a ' + label_name_dict[gt_label.item()] + ' in the image'
    #
    # sentence = 'This is a very good course'
    # label = 0

    print('----------------------------original sentence:', sentence,i)
    sentence = str.lower(sentence)
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = sentence.strip().split()

    vec = np.array(model.wv[sentence])
    print('vecotr_lem:', len(vec))

    gt_text_data = torch.tensor(vec).unsqueeze(0).unsqueeze(0).to(device)
    buchong_text = torch.tensor([0] * 524).to(device)
    gt_text_data = torch.cat((gt_text_data.view(1, -1).squeeze(), buchong_text)).unsqueeze(0).view(32, 32).unsqueeze(
        0).unsqueeze(0)

    print(gt_text_data.shape)
    print(gt_image_data.shape)
    #
    gt_data = torch.cat((gt_image_data, gt_text_data), dim=1)
    print(gt_data.shape)
    # # # # #
    # #
    # # #
    from OURS.models.vision import LeNet, weights_init1
    # # # # #
    net = LeNet().to(device)
    # # print(net)
    # # # # #
    torch.manual_seed(0)
    # # # # #
    # # # # #
    net.apply(weights_init1)
    # # # # #
    criterion = cross_entropy_for_onehot
    # # # # #
    # # # # compute original gradient
    pred = net(gt_data)
    # #
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    # # # #
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    # # #
    # # # # generate dummy data.txt and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    print('------------',dummy_data.shape)
    # # dummy_image_data = torch.randn(gt_image_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    # # #
    # # #
    optimizer1 = torch.optim.LBFGS([dummy_data, dummy_label])
    # # optimizer2 = torch.optim.LBFGS([dummy_image_data, dummy_label])
    # # #
    # # #
    history = []
    temp_loss = 0

    for iters in range(100):
        def closure():
            optimizer1.zero_grad()
            # optimizer2.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer1.step(closure)
    #     optimizer2.step(closure)
        if iters % 10 == 0:
            dummy_image_data = dummy_data.squeeze(0)[0:3].unsqueeze(0)
            dummy_text_data = dummy_data.squeeze(0)[3].squeeze(0).reshape(-1)[:500].view(5, -1)
            current_loss = closure()
            temp_loss=current_loss

            history.append(tt(dummy_image_data[0].cpu()))
            recovery_ret = ''
            with torch.no_grad():

                recovered_wordvec = dummy_text_data.view(5,100).cpu().numpy()
                for vec in recovered_wordvec:
                    temp = model.wv.most_similar(positive=[vec], topn=1)
                    recovery_ret += temp[0][0] + ' '

            # current_loss = closure()
            print(iters, "loss: %.4f" % current_loss.item(),'recovered sentence: [',recovery_ret,']')
    image_loss.append(temp_loss.item())
    recoverd_number = [i for i in image_loss if i < 0.015]
    pred_label.append(torch.argmax(dummy_label).item())
    print('temp loss:', image_loss)
    print('true label:', true_label)
    print('pred label:', pred_label)
    print('label acc:', accuracy_score(true_label, pred_label))
    print('image recu acc:', len(recoverd_number) / len(image_loss))
    with torch.no_grad():
        psnr = gt_psnr(gt_data, dummy_data, factor=1 / ds)
        print('-----------------:', psnr)
        if psnr > 0:
            psnr_list.append(psnr)
        else:
            psnr_list.append(0)
    print(psnr_list)
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    plt.show()
print('avg psnr:', np.mean(psnr_list))