import numpy as np
#import matplotlib.pyplot as plt
from skimage import io, img_as_float
#import os
#import time
import streamlit as st
from musica import *



#%% User Inputs
st.markdown("<h1 style='text-align: center; color: black;'>Interactive Image Contrast Enhancement</h1>", unsafe_allow_html=True)
  
#logoImg =  img_as_float(io.imread('/logoImage.JPG',as_gray=False))
#st.sidebar.image(logoImg)
st.sidebar.markdown("Contact: Dr. Mahesh R Panicker (mahesh@iitpkd.ac.in) ")
st.sidebar.markdown("(c) Center for Computational Imaging, IIT Palakkad")

L = st.sidebar.slider('Number of levels', 1, 7, 4)
a = np.full(L, 1)
gammaCorrFlag=st.sidebar.checkbox('Enable Gamma Correction', value=False)
if gammaCorrFlag:
    xc1 = st.sidebar.slider('Lower Intensity Limit for Gamma Correction', 0.0, 3.0, 1.0)    
    params1 = {
        'M': 1,
        'a': 1,
        'p': 1,
        'xc': xc1
        }

xc=float(st.sidebar.text_input('Lower Intensity Limit for Laplacian Pyramid Correction',0.01))
# xc = st.sidebar.slider('Lower Intensity Limit for Laplacian Pyramid Correction', 0.000, 1.000, 0.01) 
p=np.zeros((L,1))
for ii in range(L):
    if ii==0:
        p[ii]=st.sidebar.slider('p-value for level-'+str(ii), 0.0, 1.0, 0.5)
    else:
        p[ii]=st.sidebar.slider('p-value for level-'+str(ii), 0.0, 1.0, 1.0) 



params = {
        'M': 1,
        'a': a,
        'p': p,
        'xc': xc
        }

file = st.file_uploader("")  

# if st.button ('Analyze'):
    
img_o =  img_as_float(io.imread(file,as_gray=True))

img_o = (img_o-np.mean(img_o.flatten()))/(np.max(img_o.flatten())-np.min(img_o.flatten()))
if gammaCorrFlag:
    img_e=non_linear_gamma_correction(img_o, params1)
else:
    img_e=img_o
img_e = (img_e-np.mean(img_e.flatten()))/(np.max(img_e.flatten())-np.min(img_e.flatten()))

img_enhanced = musica(img_e,L,params)

img_enhanced = img_enhanced-np.mean(img_enhanced.flatten())  

imageOrgLocation = st.empty()
imageEnhLocation = st.empty()

img_o = (img_o-np.min(img_o.flatten()))/(np.max(img_o.flatten())-np.min(img_o.flatten()))
img_enhanced = (img_enhanced-np.min(img_enhanced.flatten()))/(np.max(img_enhanced.flatten())-np.min(img_enhanced.flatten()))
imageOrgLocation.image(img_o,caption='Original Image',use_column_width=True)
imageEnhLocation.image(img_enhanced,caption='Enhanced Image',use_column_width=True)

# plt.imshow(img_o,cmap="gray")
# plt.title('Original Image')
# #plt.show()

# plt.imshow(img_enhanced, cmap="gray")
# plt.title('Enhanced Image')
# #plt.show()
    
    