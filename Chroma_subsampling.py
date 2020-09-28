import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))


def show(im):
    plt.imshow(im)
    plt.show()



image = cv2.imread("C:/Users/123/Desktop/chroma_subsampling2.png")

# image = cv2.imread("C:/Users/123/Desktop/chroma_subsampling1.jpg")
width = 512
height = 512
dim = (width, height)
im = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


cv2.imshow("Original Image (4:4:4) ", im)





# im_ycrcb = rgb2ycbcr(im.copy())
# for i in range(0, len(im_ycrcb), 1):
#     for j in range(0, len(im_ycrcb[0]), 2):
#         avg = (im_ycrcb[i][j][0])
#         im_ycrcb[i][j][0] = avg
#         im_ycrcb[i][j+1][0] = avg
# im_rgb = ycbcr2rgb(im_ycrcb)
# cv2.imshow("Luminous", im_rgb)

C =  rgb2ycbcr(im.copy())
C[:, 1::2] = C[:, ::2] # 4:2:2
print("4:2:2 form ",C)
aa1 = ycbcr2rgb(C)
cv2.imshow("4:2:2 form", aa1)

D =rgb2ycbcr(im.copy())

D[1::2, :] = D[::2, :] # 4:1:0
print("4:1:0->",D)
aa2 = ycbcr2rgb(D)
cv2.imshow("4:2:1 form", aa1)

E = rgb2ycbcr(im.copy())# 4:2:0
E[1::2, :] = E[::2, :]
# Vertically, every second element equals to element above itself.
E[:, 1::2] = E[:, ::2]
# Horizontally, every second element equals to the element on its left side.
print(E)
print("4:2:0->",E)
aa3 = ycbcr2rgb(E)
cv2.imshow("4:2:0 form", aa3)

# a = rgb2ycbcr(im.copy())
# for i in range(0, len(a), 1):
#     for j in range(0, len(a[0]), 2):
#         a[i][j][1] = a[i][j + 1][1]
# aa = ycbcr2rgb(a)
# cv2.imshow("Red Chrominance", aa)

#
# b = rgb2ycbcr(im.copy())
# for i in range(0, len(b), 1):
#     for j in range(0, len(b[0]), 2):
#         b[i][j][1] = b[i][j + 1][1]
#         b[i][j][2] = b[i][j + 1][2]
# bb = ycbcr2rgb(b)
# cv2.imshow("Blue Chrominance", bb)

cv2.waitKey(0)
