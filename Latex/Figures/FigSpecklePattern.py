import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

font = {'size' : 16}
plt.rc('font', **font)

pathdados = 'e1O1_0000_0.tiff'
img0 = cv.imread(pathdados, cv.IMREAD_GRAYSCALE)  # cv.imread(pathdados, 0)
dpi = plt.rcParams['figure.dpi']
Height, Width = img0.shape

roiX1, roiY1 = 100, 200
roiX2, roiY2 = 1000, 400

img0roi = img0[roiY1:roiY2,roiX1:roiX2]

# show original image
fig = plt.figure(figsize=(10,4))
fig.subplots_adjust(top=0.8)
ax1 = fig.add_axes([0.09,0.1,0.4,0.8])
plt.set_cmap('gray')
ax1.imshow(img0, aspect="auto")
ax1.annotate('a)', xy=(3, 1),  xycoords='data', weight='bold', fontsize=22,
            xytext=(-0.2, 0.999), textcoords='axes fraction')

ax2 = fig.add_axes([0.58,0.1,0.4,0.8])
ax2.hist(img0roi.ravel(), 256, [0, 256])
ax2.annotate('intensity', xy=(3, 1),  xycoords='data',
            xytext=(0.99, 0.1), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top')
ax2.annotate('pixels', xy=(3, 1),  xycoords='data',
            xytext=(0.18, 0.995), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top')
ax2.annotate('b)', xy=(3, 1),  xycoords='data', weight='bold', fontsize=22,
            xytext=(-0.18, 0.999), textcoords='axes fraction')

plt.show()
fig.savefig("SpecklePattern.jpg", dpi=150)