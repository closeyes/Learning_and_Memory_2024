%matplotlib qt
import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
from numba import jit
img_size = 128
def preprocessing(img, w=img_size, h=img_size):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten


camera = skimage.data.camera()
astronaut = rgb2gray(skimage.data.astronaut())
celll = rgb2gray(skimage.data.cell())
coffee = rgb2gray(skimage.data.coffee())

data = [camera, astronaut, celll, coffee]

data = [preprocessing(d) for d in data]

@jit
def hopfield_train (train_set) :

    (train_img_num, num_spin) = np.shape(train_set)
    
    w = np.zeros((num_spin,num_spin))

    for img_ind in range(train_img_num) :
        print(img_ind)
        for i in range(num_spin) :
            for j in np.arange(i,num_spin) :
                
                if i==j :
                    w[i,j] += train_set[img_ind][i] * train_set[img_ind][j]
                else : 
                    w[i,j] += train_set[img_ind][i] * train_set[img_ind][j]
                    w[j,i] += train_set[img_ind][i] * train_set[img_ind][j]

    return w

w = hopfield_train(data)

noisy1 = np.random.choice(a=[-1, 0], size=img_size*img_size, p=[0.3, 0.7])
noisy2 = np.random.choice(a=[-1, 0], size=img_size*img_size, p=[0.4, 0.6])
noisy3 = np.random.choice(a=[-1, 0], size=img_size*img_size, p=[0.5, 0.5])
noisy4 = np.random.choice(a=[-1, 0], size=img_size*img_size, p=[0.6, 0.4])

cam_noise = data[0] + noisy2
astronaut_noise = data[1] + noisy3
celll_noise = data[2] + noisy1
coffee_noise = data[3] + noisy4

cam_noise[cam_noise<=0] = -1
cam_noise[cam_noise>0] = 1
astronaut_noise[astronaut_noise<=0] = -1
astronaut_noise[astronaut_noise>0] = 1
celll_noise[celll_noise<=0] = -1
celll_noise[celll_noise>0] = 1
coffee_noise[coffee_noise<=0] = -1
coffee_noise[coffee_noise>0] = 1

new=astronaut_noise + data[2]
new[new<=0] = -1
new[new>0] = 1

noisy_data = [cam_noise[:], astronaut_noise[:], celll_noise[:], coffee_noise[:]]

def retrieve (inp, w, iter) :

    res = []
    tmp = inp.reshape(img_size,img_size)
    pa = tmp.copy()
    res.append(pa)
    tmp = tmp.reshape(img_size*img_size,)

    for _ in range(iter) :

        plt.figure()
        plt.imshow(tmp.reshape(img_size,img_size))
        tmp = tmp.reshape(img_size*img_size,)
        x = np.dot(tmp, w)
        x[x>0] = 1
        x[x<=0] = -1
        tmp = x.copy()
        pp = x.copy()
        res.append(pp)
        plt.figure()
        plt.imshow(tmp.reshape(img_size,img_size))
   
        #noisy1 = np.random.choice(a=[-1, 0], size=img_size*img_size, p=[0.1, 0.9])
        mm = tmp.copy()
        mm = np.float64(mm)
        mm += np.random.randn(img_size*img_size)
        mm[mm>0] = 1
        mm[mm<=0] = -1
        mm = np.int32(mm)
        tmp = mm
        plt.figure()
        plt.imshow(tmp.reshape(img_size,img_size))
        
    
    return res

plt.figure()
iter = 3
for i in range(1) :
    
    res = retrieve(noisy_data[i], w, iter)

    for j in range(iter+1) :
        
        k = i*(iter+1) + (j+1)
        ax=plt.subplot(4,iter+1,k)
        ax.imshow(res[j].reshape(img_size,img_size))

