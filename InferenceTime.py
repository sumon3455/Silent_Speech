# PyTorch
from torchvision import transforms
from torchvision.utils import save_image 
import torch
from torch.utils.data import Dataset,DataLoader
# Data science tools
import numpy as np
import os
from skimage import io   
from timeit import default_timer as timer 
# customized functions 
from utils import *
from models import *

###############
data_dir = '/content/Data/Test/fold_1/'
model_dir = '/content/gdrive/MyDrive/KArSL/Fixed/Segmented/Results/mobilenet_v2_self_20/mobilenet_v2_self_20_fold_1.pt'
image_size = 256 
input_mean = [0.0986,0.0986,0.0986]  
input_std = [0.1844,0.1844,0.1844] 
device = 'cuda'    # 'cpu' or 'cuda' 
N = 1000 # The code will compute the inference time N times, then take the average
##############


# class TestData(Dataset): 
#     def __init__(self, root_dir, img_names, h, w, mean, std):
#         self.root_dir = root_dir
#         self.img_names = img_names
#         self.h = h # image height 
#         self.w = w # image width
#         self.mean = mean
#         self.std = std 
#         if len(mean)==3:
#             self.img_transforms = transforms.Compose([ 
#                 transforms.ToPILImage(),
#                 # transforms.Grayscale(num_output_channels=3),   
#                 transforms.Resize((self.h,self.w)),  
#                 transforms.ToTensor(), 
#                 # transforms.Normalize(mean=self.mean, std=self.std) 
#                 ]) 
#         elif len(mean)==1:
#             self.img_transforms = transforms.Compose([ 
#                 transforms.ToPILImage(),
#                 # transforms.Grayscale(num_output_channels=1),   
#                 transforms.Resize((self.h,self.w)),  
#                 transforms.ToTensor(), 
#                 # transforms.Normalize(mean=self.mean, std=self.std) 
#                 ])  
#     def __len__(self):
#         return len(self.img_names)
#     def __getitem__(self, index):
#         # read image
#         image = io.imread(os.path.join(self.root_dir, self.img_names[index]))
#         # apply transformation
#         image = self.img_transforms(image)
#         return image, self.img_names[index] 

if __name__ ==  '__main__':  
    # load model 
    checkpoint = torch.load(model_dir)
    model = checkpoint['model'] 
    categories = checkpoint['categories'] 
    del checkpoint 
    # Set to evaluation mode
    model.eval()
    # set device to 'cpu' or 'cuda'
    model = model.to(device)  
    # dataloader
    img_names = os.listdir(data_dir)
    my_transforms = transforms.Compose([ 
                transforms.ToPILImage(),
                # transforms.Grayscale(num_output_channels=1),   
                transforms.Resize((image_size,image_size)),  
                transforms.ToTensor(), 
                # transforms.Normalize(mean=self.mean, std=self.std) 
                ]) 
    # categories, n_Class_test, img_names_test, labels_test, class_to_idx, idx_to_class = Createlabels(data_dir)
    n_Class_test, img_names_test, labels_test = Retrievelabel(data_dir,categories) 
    labels_test = torch.from_numpy(labels_test).to(torch.int64)
    test_ds = MyVideosData(root_dir=data_dir,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=False,mean=input_mean,std=input_std,width=image_size,height=image_size) 
    # test_ds = TestData(root_dir=data_dir, img_names=img_names, h=image_size, w=image_size, mean=input_mean, std=input_std) 
    # test_dl = DataLoader(test_ds,batch_size=1,shuffle=False,pin_memory=True,num_workers=1)
    test_dl =  DataLoader(test_ds,batch_size=1,shuffle=False,pin_memory=True,num_workers=1) 
    # read first image 
    for data, targets, im_path in test_dl:
        if device=='cuda':
            data = data.to('cuda', non_blocking=True)
        break
    
    
    Total_time = 0.0
    for i in range(N):
        # compute network output 
        input_time = timer()
        out = model(data) 
        output_time = timer() 
        output_time = output_time - input_time
        Total_time = Total_time + output_time
        del out

    Total_time = Total_time/N 
    print(Total_time) 








