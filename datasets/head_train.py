import torch.utils.data as data
import PIL.Image as Image
import os
import torch
import numpy as np
import ipdb
from torchvision.transforms import transforms



def Getdatasetfromfile(train):  
    if train == 0:  
        PathOfData = 'xxx/data/head/image'
        PathOfTarget = 'xxx/data/head/label'
    else:
        PathOfData = None
        PathOfTarget = None
    # ImageSet = []
    ImageSet = os.listdir(PathOfData)   
    ImageSet.sort()
    MaskSet = os.listdir(PathOfTarget)
    MaskSet.sort()

    ImageSetfinal = []
    MaskSetfinal = []
    # ipdb.set_trace()
    for i in range(len(ImageSet)):
        name1 = ImageSet[i].split('.')[0]
        name2 = MaskSet[i].split('.')[0]
        if name1 == name2:
            ImageSetfinal.append(os.path.join(PathOfData, ImageSet[i]))
            MaskSetfinal.append(os.path.join(PathOfTarget, MaskSet[i]))

    data = []
    for img, lab in zip(ImageSetfinal, MaskSetfinal):
        data.append((img, lab))

    return data


class TrainDataset(data.Dataset):    
    def __init__(self, istrain, original_size, load_size):
        data = Getdatasetfromfile(train = istrain)
        self.data = data
        
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        self.prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.load_size = load_size
        self.original_size = original_size

    def __getitem__(self, index):
        x_path, y_path = self.data[index]   
       
        img_x = Image.open(x_path).convert('RGB')   
        lab_y = np.load(y_path,allow_pickle=True)
        
        lab_y = torch.Tensor(lab_y).long()    
        

        if self.load_size is not None:
            pil_image = transforms.Resize(self.load_size, interpolation=transforms.InterpolationMode.LANCZOS)(img_x)
        
        image = self.prep(pil_image)#[None,...]

        image_y = image.shape[-2]
        image_x = image.shape[-1]
        lab_y_small = torch.zeros_like(lab_y)
        lab_y_small[:,0] = lab_y[:,0] / self.original_size[0] * image_y
        lab_y_small[:,1] = lab_y[:,1] / self.original_size[1] * image_x
        lab_y_small = torch.floor(lab_y_small).long()

        
        return image, lab_y, lab_y_small, x_path  

    def __len__(self):
        return len(self.data)

