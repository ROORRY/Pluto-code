import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#from .cv2 import *
from io import BytesIO
import codecs
import time

def VideoCapture(W,H,CH,path):
    cap=cv2.VideoCapture(0)
    
    while(True):
        ret,frame=cap.read()
        cv2.imwrite(path+"/img_orgin.jpg",frame)
        if(CH == 1):
            img_path = path+"/img_transmit_gray.jpg"
            img_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_data = cv2.resize(img_orig,(W,H),interpolation=cv2.INTER_CUBIC)#640 480
            W1,H1=img_data.shape
            cv2.imwrite(img_path,img_data)
        else:
            img_path =path+"/img_transmit.jpg"
            img_data = cv2.resize(frame,(H,W),interpolation=cv2.INTER_CUBIC)#480 640
            W1,H1,CH=img_data.shape
            cv2.imwrite(img_path,img_data)
        break
    
    cap.release()
    
    threshold=10
    np.set_printoptions(threshold=np.inf)
    
    print("W=",W1,"H=",H1,"CH=",CH)
    img_data_resh=img_data.reshape(1,-1)
    img_data_array=np.array(img_data_resh)


    arr=img_data_array
    arr_shape=np.shape(arr)
    arr_size=np.size(arr)
    arr_len=len(arr)

    c=[]
    c1=[]
    c2=[]
    dc=[]
    for i in range(arr_size):
        k=format(arr[0,i], "b")
        k_b_l=len(k)
        k_c_l=8-k_b_l
        k_b='0'*k_c_l+k
        c.append(k_b)
        for n in range(0,8):
            j=int(k_b[n])
            c1.append(j)


    print(len(c1))
    path = "BIN_DATA.txt"
    c1 = ', '.join(str(i) for i in c1)
    with open(path, 'w') as f:
        f.write(str(c1))
    
    print("............................................................................D=\n")

    #cv2.namedWindow("Oringin_Dog1")
    #cv2.imshow("Oringin_Dog1",img_data)
    #cv2.imshow("np2im_Dog1",dc_img_data)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
