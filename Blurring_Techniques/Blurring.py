import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def dft(x):
    n = len(x)
    df = []
    for i in range(n):
        a = 0.0
        for j in range(n):
            a = a + x[j] * np.exp(-2j * np.pi * i * j / n)
        df.append(a / n)
    return df



def fft(x):
    n = len(x)
    if n % 2 > 0:
        print("Error")
    elif n <= 8:
        return dft(x)
    else:
        x_even = []
        x_odd = []
        for i in range(len(x)):
            if i % 2 == 0:
                x_even.append(x[i])
            else:
                x_odd.append(x[i])

        X_even = fft(np.array(x_even))
        X_odd = fft(np.array(x_odd))
        factor = np.exp(-2j * np.pi * np.arange(n) / n)

        ans = []
        for i in range(int(len(x))):
            if i < int(len(x)/2):
                ans.append(X_even[i]+factor[i]*X_odd[i])
            else:
                ans.append(X_even[i-int(n/2)]+factor[i]*X_odd[i-int(n/2)])

        return ans

def fft2d(x):
    row = len(x)
    col = len(x[0])
    a = []
    for i in range(row):
         a.append(fft(x[i]))

    b = []

    for j in range(col):
        temp = []
        for i in range(row):
            temp.append((a[i][j]))
        b.append(fft(temp))

    for i in range(len(b)):
        for j in range(len(b[0])):
            if i < j :
                t = b[i][j]
                b[i][j] = b[j][i]
                b[j][i] = t

    return b



def idft(x):
    n = len(x)
    df = []
    for i in range(n):
        a = 0.0
        for j in range(n):
            a = a + x[j] * np.exp(2j * np.pi * i * j / n)
        df.append(a / n)
    return df


def ifft(x):
    n = len(x)
    if n % 2 > 0:
        print("Error")
    elif n<= 8:
        return idft(x)
    else:
        x_even = []
        x_odd = []
        for i in range(len(x)):
            if i%2==0 :
                x_even.append(x[i])
            else:
                x_odd.append(x[i])
        X_even = ifft(np.array(x_even))
        X_odd = ifft(np.array(x_odd))
        factor = np.exp(2j * np.pi * np.arange(n) / n)

        ans = []
        for i in range(int(len(x))):
            if i < int(len(x)/2) :
                ans.append(X_even[i]+factor[i]*X_odd[i])
            else:
                ans.append(X_even[i-int(n/2)]+factor[i]*X_odd[i-int(n/2)])

        return ans


def ifft2d(x):
    row = len(x)
    l = x[0]
    col = len(l)
    a = []
    for i in range(row):
         a.append(ifft(x[i]))
    b = []
    for j in range(col):
        temp = []
        for i in range(row):
            temp.append((a[i][j]))
        b.append(ifft(temp))

    for i in range(len(b)):
        for j in range(len(b[0])):
            if i < j :
                t = b[i][j]
                b[i][j] = b[j][i]
                b[j][i] = t

    b  = np.array(b)/(row*col)

    return b

def fft2d_kernel(x):
    row = len(x)
    col = len(x[0])
    a = []
    for i in range(row):
         a.append(dft(x[i]))
    b = []
    for j in range(col):
        temp = []
        for i in range(row):
            temp.append((a[i][j]))
        b.append(dft(temp))
    for i in range(len(b)):
        for j in range(len(b[0])):
            if i < j:
                t = b[i][j]
                b[i][j] = b[j][i]
                b[j][i] = t

    return b



def duv(u, v):
    return np.sqrt(u*u+v*v)


def convolution(input_image, h):

    g = input_image
    for i in range(2, len(input_image)-2):
        for j in range(2, len(input_image[0])-2):
            g[i][j] = 0
            for x in range(-2,  3):
                for y in range(-2, 3):
                    g[i][j] = g[i][j] + input_image[i+x][j+y]*h[2+x][2+y]
    return g


def get_kernel():
    d = 150
    size = 256
    a = []
    for i in range(size):
        temp = []
        for j in range(size):
            if duv(i-size/2, j-size/2) > d:
                temp.append(1)
            else:
                temp.append(0)
        a.append(temp)
    return a


image = cv2.imread("C:/Users/123/Desktop/Blurr_Example.png", 0)

cv2.imshow("Input Image", image)

start_time = time.time()

image_fft = fft2d(image.copy())

end_time = time.time()
print("Time for fft2d :")
print(end_time-start_time)



#To show the inbulit FFT and Our FFT are nearly Same
fshift = np.fft.fftshift(image_fft.copy())
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(211), plt.imshow(magnitude_spectrum, 'gray')
plt.title('Fourier Transform of Input Image'), plt.xticks([]), plt.yticks([])

fshift = np.fft.fftshift(np.fft.fft2(image.copy()))
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(212), plt.imshow(magnitude_spectrum, 'gray')
plt.title('Fourier Transform of Input Image By Inbuilt Funtion'), plt.xticks([]), plt.yticks([])
plt.show()


n = get_kernel()

output = image_fft
for i in range(256):
    for j in range(256):
        output[i][j] = image_fft[i][j]*n[i][j]


start_time = time.time()

o = ifft2d(output)

end_time = time.time()
print("Time for ifft2d :")
print(end_time-start_time)

# Taking kernel for convolution
k = []
a = [1/25, 1/25, 1/25, 1/25, 1/25]
k.append(a)
k.append(a)
k.append(a)
k.append(a)
k.append(a)

start_time = time.time()

convol_image = convolution(image.copy(), k.copy())

end_time = time.time()
print("convolution time: ")
print(end_time-start_time)


plt.imshow(np.abs(o), 'gray')
plt.title('Blurring in Frequency Domain'), plt.xticks([]), plt.yticks([])

cv2.imshow("Blurring in Spatial Domain", convol_image)
plt.show()
cv2.waitKey(0)
