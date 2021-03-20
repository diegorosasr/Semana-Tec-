#By Sherpa

#code taken from https://www.kaggle.com/thesherpafromalabama/manual-image-convolution-zero-padding

#Modified by: Diego Rosas

import numpy as np
from functools import reduce
import scipy.signal
from skimage import io
import matplotlib.pyplot as plt

def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)            
    
    # Add zero padding to the input image 
    image_padded = np.zeros((image.shape[0] + (kernel.shape[0]-1), 
                             image.shape[1] + (kernel.shape[1]-1)))   
    image_padded[(kernel.shape[0]//2):-(kernel.shape[0]//2), 
                 (kernel.shape[1]//2):-(kernel.shape[1]//2)] = image
    
    for x in range(image.shape[1]):     
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output

img = io.imread('../input/peppers.tif')  

print('image matrix size: ', img.shape )     

plt.imshow(img, cmap=plt.cm.gray) 
plt.axis('off')
plt.show()

kernel = np.ones([7,7])  
k_elements = reduce(lambda x, y: x * y, np.shape(kernel))
kernel = kernel/k_elements
cnvlvd_img = convolve2d(img,kernel)
print('\n First 5 columns and rows of the standar average matrix: \n', cnvlvd_img[:5,:5]*255)

plt.imshow(cnvlvd_img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
    return output
