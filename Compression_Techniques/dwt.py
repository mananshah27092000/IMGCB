import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from PIL import Image

high = [-0.04, 0.08, -0.04]
# high = [-0.05, 0.1, -0.05]
low = [0.25, 0.5, 0.25]
# low = [0.04, 0.92, 0.04]

def swap(x3):
    for i in range(len(x3)):
        for j in range(len(x3[0])):
            if i < j:
                temp = x3[i][j]
                x3[i][j] = x3[j][i]
                x3[j][i] = temp
    return x3

def convolve(x,v):
    h = v.copy()
    z = x.copy()
    h = h[::-1]
    z = [0]+[0]+z+[0]+[0]
    x5 = []
    for i in range(2,len(z)):
        sum = 0
        l = 2
        for j in range(len(h)):
            sum = sum + z[i-j] * h[l]
            l = l - 1
        x5.append(sum)
    return x5


def convolution_row(x3,h):
    z = []
    for t in range(len(x3)):
        z.clear()
        for i in range(1, (len(x3[t]))-1):
            z.append(x3[t][i])
        x3[t] = (convolve(z, h))
    return x3


def down_sample(x3):
    for k in range(len(x3)):
        for t in range(len(x3)):
            if (2*t) < len(x3):
                x3[k][t] = x3[k][(2*t)]
            else:
                x3[k][t] = 0
    return x3

def LL(x3):
    x3 = convolution_row(x3, low)
    x3 = down_sample(x3)
    x3 = swap(x3)
    x3 = convolution_row(x3, low)
    x3 = down_sample(x3)
    x3 = swap(x3)
    # cv2.imshow("Low Low", x3)

def HL(x3):
    x3 = convolution_row(x3, high)
    x3 = down_sample(x3)
    x3 = swap(x3)
    x3 = convolution_row(x3, low)
    x3 = down_sample(x3)
    x3 = swap(x3)
    # cv2.imshow("High Low", x3)

def LH(x3):
    x3 = convolution_row(x3, low)
    x3 = down_sample(x3)
    x3 = swap(x3)
    x3 = convolution_row(x3, high)
    x3 = down_sample(x3)
    x3 = swap(x3)
    # cv2.imshow("Low High", x3)

def HH(x3):
    x3 = convolution_row(x3, high)
    x3 = down_sample(x3)
    x3 = swap(x3)
    x3 = convolution_row(x3, high)
    x3 = down_sample(x3)
    x3 = swap(x3)
    # cv2.imshow("High High", x3)

oog = pywt.data.camera()
org = pywt.data.camera()
org_LL = pywt.data.camera()
org_LL2 = pywt.data.camera()
org_LL3 = pywt.data.camera()
i1 = pywt.data.camera()
i2 = pywt.data.camera()
i3 = pywt.data.camera()
i4 = pywt.data.camera()
i5 = pywt.data.camera()
i6 = pywt.data.camera()
i7 = pywt.data.camera()
output = pywt.data.camera()

scale_percent = 50
width = int(oog.shape[1] * scale_percent / 100)
height = int(oog.shape[0] * scale_percent / 100)
dim = (width, height)
resizedLL = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedLH = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedHL = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedHH = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
LL(i1)
LL(org_LL)
LL(org_LL2)
LL(org_LL3)
LL(i5)
LL(i6)
LL(i7)
HL(i2)
LH(i3)
HH(i4)

LL(i1)
HL(i5)
LH(i6)
HH(i7)

for i in range(len(resizedLL)):
    for j in range(len(resizedLL)):
        resizedLH[i][j] = i3[i][j]
        resizedHL[i][j] = i2[i][j]
        resizedHH[i][j] = i4[i][j]
        resizedLL[i][j] = org_LL[i][j]

scale_percent = 25
width = int(oog.shape[1] * scale_percent / 100)
height = int(oog.shape[0] * scale_percent / 100)
dim = (width, height)
resizedLL1 = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedLH1 = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedHL1 = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)
resizedHH1 = cv2.resize(oog, dim, interpolation=cv2.INTER_AREA)

for i in range(len(resizedLL1)):
    for j in range(len(resizedLL1)):
        resizedLH1[i][j] = i6[i][j]
        resizedHL1[i][j] = i5[i][j]
        resizedHH1[i][j] = i7[i][j]
        resizedLL1[i][j] = i1[i][j]
# LLLL = i1
# LLLH = i6
# LLHL = i5
# LLHH = i7
# LH = i3
# HL = i2
# HH = i4
cv2.waitKey(0)
titles = ['LLLL', 'LLLH',
          'LLHL', 'LLHH', 'LL', 'LH',
          'HL', 'HH']
fig = plt.figure(figsize=(12, 12))
for i, a in enumerate([resizedLL1, resizedLH1, resizedHL1, resizedHH1, resizedLL, resizedLH, resizedHL, resizedHH]):
    ax = fig.add_subplot(2, 4, i + 1)
    ax.imshow(a, plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
for i in range(512):
    for j in range(512):
        if i<128 and j<128:
            output[i][j] = i1[i][j]
        elif i<256 and j<128:
            output[i][j] = i5[i-128][j]
        elif i<2*64 and j<256:
            output[i][j] = i6[i][j-2*64]
        elif i<2*128 and j<2*128:
            output[i][j] = i7[i-2*64][j-2*64]
        elif i<2*128:
            output[i][j] = i3[i][j-2*128]
        elif j<2*128:
            output[i][j] = i2[i-2*128][j]
        else:
            output[i][j] = i4[i-2*128][j-2*128]
        if i == 128 and j<256:
            output[i][j] = 255
        if j == 128 and i<256:
            output[i][j] = 255
        if i == 256 or j == 256:
            output[i][j] = 255
# cv2.imwrite('D:/final_result.jpg', output)
cv2.imshow("org_ll", output)
cv2.waitKey(0)

def up_sample(x3,target):
    x5 = target
    for k in range(len(x5)):
        for t in range(len(x5)):
            if t % 2 == 0 and k < len(x3):
                x5[k][t] = x3[k][(t//2)]
            elif k < len(x3):
                x5[k][t] = 0
    return x5

def re_convolution_row(x3,h):
    z = []
    for t in range(len(x3)//2):
        z.clear()
        for i in range(1, (len(x3[t]))-1):
            z.append(x3[t][i])
        x3[t] = (convolve(z, h))
    return x3


def re_HH(x3):
    x3 = swap(x3)
    org_LS1 = pywt.data.camera()
    LL(org_LS1)
    x3 = up_sample(x3, org_LS1)
    x3 = re_convolution_row(x3, high)
    x3 = swap(x3)
    org_LS = pywt.data.camera()
    LL(org_LS)
    x3 = up_sample(x3, org_LS)
    x3 = convolution_row(x3, high)
    return x3

def re_HL(x3):
    x3 = swap(x3)
    org_LS1 = pywt.data.camera()
    LL(org_LS1)
    x3 = up_sample(x3, org_LS1)
    x3 = re_convolution_row(x3, low)
    x3 = swap(x3)
    org_LS = pywt.data.camera()
    LL(org_LS)
    x3 = up_sample(x3, org_LS)
    x3 = convolution_row(x3, high)
    return x3

def re_LH(x3):
    x3 = swap(x3)
    org_LS1 = pywt.data.camera()
    LL(org_LS1)
    x3 = up_sample(x3, org_LS1)
    x3 = re_convolution_row(x3, high)
    x3 = swap(x3)
    org_LS = pywt.data.camera()
    LL(org_LS)
    x3 = up_sample(x3, org_LS)
    x3 = convolution_row(x3, low)
    return x3

def re_LL(x3):
    x3 = swap(x3)
    org_LS1 = pywt.data.camera()
    LL(org_LS1)
    x3 = up_sample(x3, org_LS1)
    x3 = re_convolution_row(x3, low)
    x3 = swap(x3)
    org_LS = pywt.data.camera()
    LL(org_LS)
    x3 = up_sample(x3, org_LS)
    x3 = convolution_row(x3, low)
    return x3

xLLLL = re_LL(i1)
xLLLH = re_LH(i6)
xLLHL = re_HL(i5)
xLLHH = re_HH(i7)

xLL = re_LL(resizedLL)
xLH = re_LH(i3)
xHL = re_HL(i2)
xHH = re_HH(i4)

# cv2.imshow('Reconstructed', xLH)


cv2.waitKey(0)

