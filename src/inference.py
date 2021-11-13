from model import ResNetUNet
import torch
from skimage.segmentation import mark_boundaries
from scipy import ndimage as ndi
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A

#Model Loading
model_1 = ResNetUNet().to('cuda')
model_1.load_state_dict(state_dict=torch.load("/media/brats/DRIVE1/akansh/lung-seg/notebook/weights/0.977697_.pth")['state_dict'])

#helper function
def show_img_mask(img, mask):    
    img_mask = mark_boundaries(np.array(img),mask.astype('bool'),outline_color=(0,1,0),color=(0,1,0))
    plt.imshow(img_mask)


def lung_seg(img_path, model = model_1, back = False, plot = True):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = np.copy(img)
    
    norm = A.Normalize()
    img = norm(image = img)['image']
    DEVICE = 'cuda'
    model.eval()
    with torch.no_grad():
        out = model(torch.unsqueeze(torch.tensor(img), dim = 0).permute(0,3,1,2).to(DEVICE).float())
        out_2 = np.copy(out.data.cpu().numpy())
        out_2[np.nonzero(out_2 < 0.5)] = 0.0
        out_2[np.nonzero(out_2 >= 0.5)] = 1.0
        
    if plot == True:
        plt.figure(figsize=(10,15))
        plt.subplot(1, 3, 1)
        plt.imshow(resized, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.imshow(out_2[0][0], cmap="gray")

        plt.subplot(1, 3, 3)
        show_img_mask(resized,out_2[0][0])
    
    if back == True:
        return resized, out_2[0][0]

path = input("Please Enter the path of Image: ")
orig, mask = lung_seg(str(path), back = True, plot=False)
cv2.imwrite("result-mask.png", mask*255)