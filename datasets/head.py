import torchvision.transforms as transforms
import numpy as np
import os, random, math, torch
import torch.utils.data as data
import json
from PIL import Image
from .augment import cc_augment, augment_patch, np_gray_to_PIL, gray_to_PIL

from imgaug.augmentables import KeypointsOnImage

import numpy as np
import imgaug.augmenters as iaa
from skimage import io
import ipdb

import cv2
import numpy as np
from matplotlib import pyplot as plt
from albumentations import Compose, Resize, RandomBrightnessContrast, RandomResizedCrop, ShiftScaleRotate, VerticalFlip, HorizontalFlip, ElasticTransform
import torch
import albumentations

class Head_Base(data.Dataset):
    def __init__(self, pathDataset, orgignal_size=[2400, 1935], size=[384, 384], mode='Train', id_shot=None):
    
        self.original_size = orgignal_size
        self.size = size
        self.list = list()
        self.num_landmark = 19

        self.pixel_spaceing = 0.1

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))      
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        if mode == 'Oneshot':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            assert id_shot is not None
            start = id_shot
            end = id_shot
        elif mode == 'Train':  
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Infer_Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
            
        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        self.images = {item['ID']: Image.open(os.path.join(self.pth_Image, item['ID']+'.bmp')).convert('RGB')\
            for item in self.list}  
        
        self.path_list =  {item['ID']: os.path.join(self.pth_Image, item['ID']+'.bmp')\
            for item in self.list}

        self.mode = mode
    
    def resize_landmark_dataset(self, landmark):  
        res = [0] * len(landmark)
        res[1] = int(landmark[0] * self.size[1] / self.original_size[1]) # x
        res[0] = int(landmark[1] * self.size[0] / self.original_size[0]) # y
        return res

    def compute_spacing(self, pixel_spacing_x, pixel_spacing_y=None):  
        if pixel_spacing_y is None:
            pixel_spacing_y = pixel_spacing_x
        
        scale_rate_x = pixel_spacing_x * self.original_size[1] / self.size[1]
        scale_rate_y = pixel_spacing_y * self.original_size[0] / self.size[0]

        return [scale_rate_y, scale_rate_x]
        
    def get_landmark_gt(self, id_str):
        landmark_list = list()

        with open(os.path.join(self.pth_label_junior, id_str+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, id_str+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark_dataset(landmark))   
        
        return landmark_list

    def __getitem__(self, index: int):
        pass

    def __len__(self):

        return len(self.list)


class Head_SSL_Train(Head_Base):
    def __init__(self, pathDataset, mode, size=[384, 384], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None):
        super().__init__(pathDataset=pathDataset, size=size)

        self.min_prob = min_prob 
        self.use_probmap = use_probmap
        self.size = size
        self.patch_size = int(0.5 * self.size[0])
        # self.patch_size = 256
        self.tag_ssl = tag_ssl
        
        self.Radius = int(max(size) * radius_ratio)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)  
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    guassian_mask[i][j] = math.exp(- 0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))  
        self.guassian_mask = guassian_mask

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)  


        num_repeat = 10   
        if mode == 'Train':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.mode = mode

    def gen_probmap(self, id):
        # Load pred_landmarks
        assert(self.tag_ssl is not None)
        prob_map = torch.zeros(self.size).float()
        with open(os.path.join('xxx/final_runs', self.tag_ssl, 'ssl_scratch', 'pseudo_labels', f'{id}.json'), 'rb') as f:
            pred_landmark = json.load(f)
        for i in range(len(pred_landmark.keys())):
            margin_x_left = max(0, pred_landmark[str(i)][1] - self.Radius)
            margin_x_right = min(self.size[1], pred_landmark[str(i)][1] + self.Radius)
            margin_y_bottom = max(0, pred_landmark[str(i)][0] - self.Radius)
            margin_y_top = min(self.size[0], pred_landmark[str(i)][0] + self.Radius)
            prob_map[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] += \
                self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        return torch.clamp(prob_map, self.min_prob, 1)


    def __getitem__(self, index):
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
            # print("[debug] ", item['image'].shape)
        
        size = self.size
        
        # Crop 192 x 192 Patch
        patch_size = self.patch_size

        margin_x = np.random.randint(0, size[1] - patch_size)
        margin_y = np.random.randint(0, size[0] - patch_size)

        chosen_x_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))
        chosen_y_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))
        raw_y, raw_x = chosen_y_raw+margin_y, chosen_x_raw+margin_x

        
        if self.use_probmap:  
            probmap = self.gen_probmap(item['ID'])
            # gray_to_PIL(probmap).save('probmap.jpg')
            while(probmap[raw_y, raw_x] < random.random()):
                # print(f'Repeat sampling .... {probmap[raw_y, raw_x]}')
                margin_x = np.random.randint(0, size[1] - patch_size)  
                margin_y = np.random.randint(0, size[0] - patch_size)
                chosen_x_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))  
                chosen_y_raw = np.random.randint(int(0.1*patch_size), int(0.9*patch_size))
                raw_y, raw_x = chosen_y_raw+margin_y, chosen_x_raw+margin_x  

        crop_imgs = augment_patch(item['image']\
            [:, margin_y:margin_y+patch_size, margin_x:margin_x+patch_size],\
                self.aug_transform)  

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_raw, chosen_x_raw] = 1   
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]  
        # print(chosen_y, chosen_x)

        chosen_y, chosen_x = temp.argmax() // patch_size, temp.argmax() % patch_size   

        return item['image'], crop_imgs, chosen_y, chosen_x, raw_y, raw_x


class Head_SSL_Infer_SSLv1_generate(Head_Base):  # for data generation
    def __init__(self, pathDataset, mode, size=[384, 384], id_oneshot=1, load_size = [224, 224]):
        super().__init__(pathDataset=pathDataset, size=size, mode=mode, id_shot=id_oneshot)
        self.size = size
        self.var = 1.
        self.mean = (0.485, 0.456, 0.406) 
        self.std = (0.229, 0.224, 0.225)
        self.prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.load_size = load_size

    def make_heatmap(self, landmark):
        length, width = self.size
        var = self.var
        x,y = torch.meshgrid(torch.arange(0, length),
                            torch.arange(0,width))
        p = torch.stack([x,y], dim=2).float()
        inner_factor = -1/(2*(var**2))
        mean = torch.as_tensor(landmark).float()
        heatmap = (p-mean).pow(2).sum(dim=-1)
        heatmap= torch.exp(heatmap*inner_factor)
        return heatmap
    
    def image_aug(self, image_path, landmark):
        image = cv2.imread(image_path)
        size = self.size
        keypoint = landmark
        if len(keypoint) == 19:
            hm1 = self.make_heatmap(keypoint[0])
            hm2 = self.make_heatmap(keypoint[1])
            hm3 = self.make_heatmap(keypoint[2])
            hm4 = self.make_heatmap(keypoint[3])
            hm5 = self.make_heatmap(keypoint[4])
            hm6 = self.make_heatmap(keypoint[5])
            hm7 = self.make_heatmap(keypoint[6])
            hm8 = self.make_heatmap(keypoint[7])
            hm9 = self.make_heatmap(keypoint[8])
            hm10 = self.make_heatmap(keypoint[9])
            hm11 = self.make_heatmap(keypoint[10])
            hm12 = self.make_heatmap(keypoint[11])
            hm13 = self.make_heatmap(keypoint[12])
            hm14 = self.make_heatmap(keypoint[13])
            hm15 = self.make_heatmap(keypoint[14])
            hm16 = self.make_heatmap(keypoint[15])
            hm17 = self.make_heatmap(keypoint[16])
            hm18 = self.make_heatmap(keypoint[17])
            hm19 = self.make_heatmap(keypoint[18])
            hm = np.stack([hm1, hm2, hm3, hm4, hm5, hm6, hm7, hm8, hm9, hm10, hm11, hm12, hm13, hm14, hm15, hm16, hm17, hm18, hm19], axis=2)
        original_height, original_width = image.shape[:2]
        
        aug = Compose([
            ShiftScaleRotate(shift_limit=0.02, scale_limit=0.01,rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=False, p=1.0),
                            ])
        augmented = aug(image=image,mask =hm)#, keypoints=keypoint.tolist())
        image_scaled = augmented['image']
        mask_scaled = augmented['mask']
        if len(keypoint) == 19:
            x1 = np.where(mask_scaled[:, :, 0] == mask_scaled[:, :, 0].max())[0][0]
            y1 = np.where(mask_scaled[:, :, 0] == mask_scaled[:, :, 0].max())[1][0]
            x2 = np.where(mask_scaled[:, :, 1] == mask_scaled[:, :, 1].max())[0][0]
            y2 = np.where(mask_scaled[:, :, 1] == mask_scaled[:, :, 1].max())[1][0]
            x3 = np.where(mask_scaled[:, :, 2] == mask_scaled[:, :, 2].max())[0][0]
            y3 = np.where(mask_scaled[:, :, 2] == mask_scaled[:, :, 2].max())[1][0]
            x4 = np.where(mask_scaled[:, :, 3] == mask_scaled[:, :, 3].max())[0][0]
            y4 = np.where(mask_scaled[:, :, 3] == mask_scaled[:, :, 3].max())[1][0]
            x5 = np.where(mask_scaled[:, :, 4] == mask_scaled[:, :, 4].max())[0][0]
            y5 = np.where(mask_scaled[:, :, 4] == mask_scaled[:, :, 4].max())[1][0]
            x6 = np.where(mask_scaled[:, :, 5] == mask_scaled[:, :, 5].max())[0][0]
            y6 = np.where(mask_scaled[:, :, 5] == mask_scaled[:, :, 5].max())[1][0]
            x7 = np.where(mask_scaled[:, :, 6] == mask_scaled[:, :, 6].max())[0][0]
            y7 = np.where(mask_scaled[:, :, 6] == mask_scaled[:, :, 6].max())[1][0]
            x8 = np.where(mask_scaled[:, :, 7] == mask_scaled[:, :, 7].max())[0][0]
            y8 = np.where(mask_scaled[:, :, 7] == mask_scaled[:, :, 7].max())[1][0]
            x9 = np.where(mask_scaled[:, :, 8] == mask_scaled[:, :, 8].max())[0][0]
            y9 = np.where(mask_scaled[:, :, 8] == mask_scaled[:, :, 8].max())[1][0]
            x10 = np.where(mask_scaled[:, :, 9] == mask_scaled[:, :, 9].max())[0][0]
            y10 = np.where(mask_scaled[:, :, 9] == mask_scaled[:, :, 9].max())[1][0]
            x11 = np.where(mask_scaled[:, :, 10] == mask_scaled[:, :, 10].max())[0][0]
            y11 = np.where(mask_scaled[:, :, 10] == mask_scaled[:, :, 10].max())[1][0]
            x12 = np.where(mask_scaled[:, :, 11] == mask_scaled[:, :, 11].max())[0][0]
            y12 = np.where(mask_scaled[:, :, 11] == mask_scaled[:, :, 11].max())[1][0]
            x13 = np.where(mask_scaled[:, :, 12] == mask_scaled[:, :, 12].max())[0][0]
            y13 = np.where(mask_scaled[:, :, 12] == mask_scaled[:, :, 12].max())[1][0]
            x14 = np.where(mask_scaled[:, :, 13] == mask_scaled[:, :, 13].max())[0][0]
            y14 = np.where(mask_scaled[:, :, 13] == mask_scaled[:, :, 13].max())[1][0]
            x15 = np.where(mask_scaled[:, :, 14] == mask_scaled[:, :, 14].max())[0][0]
            y15 = np.where(mask_scaled[:, :, 14] == mask_scaled[:, :, 14].max())[1][0]
            x16 = np.where(mask_scaled[:, :, 15] == mask_scaled[:, :, 15].max())[0][0]
            y16 = np.where(mask_scaled[:, :, 15] == mask_scaled[:, :, 15].max())[1][0]
            x17 = np.where(mask_scaled[:, :, 16] == mask_scaled[:, :, 16].max())[0][0]
            y17 = np.where(mask_scaled[:, :, 16] == mask_scaled[:, :, 16].max())[1][0]
            x18 = np.where(mask_scaled[:, :, 17] == mask_scaled[:, :, 17].max())[0][0]
            y18 = np.where(mask_scaled[:, :, 17] == mask_scaled[:, :, 17].max())[1][0]
            x19 = np.where(mask_scaled[:, :, 18] == mask_scaled[:, :, 18].max())[0][0]
            y19 = np.where(mask_scaled[:, :, 18] == mask_scaled[:, :, 18].max())[1][0]
            
            labnew = [[x1, y1], [x2, y2], [x3, y3], [x4, y4],[x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9], [x10, y10], [x11, y11], [x12, y12],[x13, y13], [x14, y14], [x15, y15], [x16, y16], [x17, y17], [x18, y18], [x19, y19]]

        image = Image.fromarray(cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB))

        return image, labnew

    def __getitem__(self, index):
        item = self.list[index]

        

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark_dataset(landmark))

        scale_rate = self.compute_spacing(pixel_spacing_x=self.pixel_spaceing)


        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        image, landmark_list = self.image_aug(pth_img, landmark_list)

        

        return image, landmark_list, pth_img   

    def __len__(self):

        return len(self.list)

class Head_SSL_Infer(Head_Base):
    def __init__(self, pathDataset, mode, size=[384, 384], id_oneshot=1):
        super().__init__(pathDataset=pathDataset, size=size, mode=mode, id_shot=id_oneshot)
        self.size = size


    def __getitem__(self, index):

        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        landmark_list = list()
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    landmark_list.append(self.resize_landmark_dataset(landmark))

        scale_rate = self.compute_spacing(pixel_spacing_x=self.pixel_spaceing)

        # if self.mode != 'Oneshot':
        #     return item['image'], landmark_list, pth_img

        return item['image'], landmark_list, pth_img   

    def __len__(self):

        return len(self.list)

class Head_TPL_Voting(Head_Base):
    def __init__(self, pathDataset, mode, size=[800, 640], do_repeat=True, ssl_dir=None, R_ratio=0.05, pseudo=True, id_shot=1):
        super().__init__(pathDataset=pathDataset, size=size, mode=mode, id_shot=id_shot)

        self.ssl_dir = ssl_dir
        self.do_repeat = do_repeat
        self.Radius = int(max(size) * R_ratio)
        self.pseudo = pseudo

        # gen mask
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    # for guassian mask
                    guassian_mask[i][j] = math.exp(- 0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.mask = mask
        self.guassian_mask = guassian_mask

        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        if mode == 'Train':
            transforms.ColorJitter(brightness=0.25, contrast=0.35)
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        self.transform = transforms.Compose(transformList)
    
        num_repeat = 9 * 1
        if mode == 'Train' and do_repeat:
            temp = self.list.copy() 
            for _ in range(num_repeat):
                self.list.extend(temp)



    def __getitem__(self, index):
        item = self.list[index]

        # if self.transform != None:
        #     pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        #     image = self.transform(Image.open(pth_img).convert('RGB'))
        image = self.images[item['ID']]
        image = self.transform(image)
        
        landmark_list = list()

        scale_rate = self.compute_spacing(pixel_spacing_x=self.pixel_spaceing)

        if self.mode == 'Train' and self.do_repeat and self.pseudo:
            with open('{0}/pseudo_labels/{1}.json'.format(self.ssl_dir, item['ID']), 'r') as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)
        else:
            with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
                with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                    for i in range(self.num_landmark):
                        landmark1 = f1.readline().split()[0].split(',')
                        landmark2 = f2.readline().split()[0].split(',')
                        landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                        landmark_list.append(self.resize_landmark_dataset(landmark))

        # GT, mask, offset
        y, x = image.shape[-2], image.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            margin_x_left = max(0, landmark[1] - self.Radius)
            margin_x_right = min(x, landmark[1] + self.Radius)
            margin_y_bottom = max(0, landmark[0] - self.Radius)
            margin_y_top = min(y, landmark[0] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        return image, mask, offset_y, offset_x, landmark_list, item['ID'], scale_rate

class Head_TPL_Heatmap(Head_Base):
    def __init__(self, pathDataset, mode, size=[800, 640], do_repeat=True, ssl_dir=None, R_ratio=0.04, pseudo=True):
        super().__init__(pathDataset=pathDataset, size=size, mode=mode)

        self.ssl_dir = ssl_dir
        self.do_repeat = do_repeat
        self.Radius = int(max(size) * R_ratio)
        self.pseudo = pseudo

        # gen mask
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    # for guassian mask
                    guassian_mask[i][j] = math.exp(- math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.mask = mask
        self.guassian_mask = guassian_mask

        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        if mode == 'Train':
            transforms.ColorJitter(brightness=0.25, contrast=0.35)
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        self.transform = transforms.Compose(transformList)
    
        num_repeat = 9 * 1
        if mode == 'Train' and do_repeat:
            temp = self.list.copy() 
            for _ in range(num_repeat):
                self.list.extend(temp)



    def __getitem__(self, index):
        item = self.list[index]

        # if self.transform != None:
        #     pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        #     image = self.transform(Image.open(pth_img).convert('RGB'))
        image = self.images[item['ID']]
        image = self.transform(image)
        
        landmark_list = list()

        scale_rate = self.compute_spacing(pixel_spacing_x=self.pixel_spaceing)

        if self.mode == 'Train' and self.do_repeat and self.pseudo:
            with open('{0}/pseudo_labels/{1}.json'.format(self.ssl_dir, item['ID']), 'r') as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)
        else:
            with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
                with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                    for i in range(self.num_landmark):
                        landmark1 = f1.readline().split()[0].split(',')
                        landmark2 = f2.readline().split()[0].split(',')
                        landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                        landmark_list.append(self.resize_landmark_dataset(landmark))

        # GT, mask, offset
        y, x = image.shape[-2], image.shape[-1]
        guassian_mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            margin_x_left = max(0, landmark[1] - self.Radius)
            margin_x_right = min(x, landmark[1] + self.Radius)
            margin_y_bottom = max(0, landmark[0] - self.Radius)
            margin_y_top = min(y, landmark[0] + self.Radius)

            guassian_mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        return image, guassian_mask, offset_y, offset_x, landmark_list, item['ID'], scale_rate

class Head_ERE(Head_Base):
    def __init__(self, cfg, mode, size=[800, 640], do_repeat=True, ssl_dir=None, R_ratio=0.04, pseudo=True, id_shot=None):
        super().__init__(pathDataset=cfg.dataset.dataset_pth, size=size, mode=mode, id_shot=id_shot)

        self.ssl_dir = ssl_dir
        self.do_repeat = do_repeat
        self.Radius = int(max(size) * R_ratio)
        self.pseudo = pseudo

        self.perform_augmentation = True if mode == 'Train' else False

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        self.transform = transforms.Compose(transformList)
        
        # Define augmentation
        data_aug_params = cfg.dataset.AUGMENTATION
        self.augmentation = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-data_aug_params.TRANSLATION_X, data_aug_params.TRANSLATION_X),
                                     "y": (-data_aug_params.TRANSLATION_Y, data_aug_params.TRANSLATION_Y)},
                       scale=[1 - data_aug_params.SF, 1],
                       rotate=[-data_aug_params.ROTATION_FACTOR, data_aug_params.ROTATION_FACTOR]),
            iaa.Multiply(mul=(1 - data_aug_params.INTENSITY_FACTOR, 1 + data_aug_params.INTENSITY_FACTOR)),
            iaa.GammaContrast(),
            iaa.ElasticTransformation(alpha=(0, data_aug_params.ELASTIC_STRENGTH),
                                      sigma=data_aug_params.ELASTIC_SMOOTHNESS, order=3)
        ])

    def __getitem__(self, index):
        item = self.list[index]

        # if self.transform != None:
        #     pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        #     image = self.transform(Image.open(pth_img).convert('RGB'))
        image = self.images[item['ID']]
        image = self.transform(image)
        image = image[0].numpy()
        
        landmark_list = list()

        scale_rate = self.compute_spacing(pixel_spacing_x=self.pixel_spaceing)

        if self.mode == 'Train' and self.do_repeat and self.pseudo:
            with open('{0}/pseudo_labels/{1}.json'.format(self.ssl_dir, item['ID']), 'r') as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)
        else:
            with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
                with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                    for i in range(self.num_landmark):
                        landmark1 = f1.readline().split()[0].split(',')
                        landmark2 = f2.readline().split()[0].split(',')
                        landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                        landmark_list.append(self.resize_landmark_dataset(landmark))
        
        if self.perform_augmentation:
        # if False:
            tmp = np.concatenate(landmark_list).reshape(-1, 2)[:, ::-1]
            kps = KeypointsOnImage.from_xy_array(tmp, shape=image.shape)
            image, kps_augmented = self.augmentation(image=image, keypoints=kps)
            # np_gray_to_PIL(image).save('test.png')
            # import ipdb; ipdb.set_trace()
            kps = kps_augmented
            
            landmark_list = np.round(kps.to_xy_array()[:,::-1]).astype(np.int32).tolist()

        # GT, mask, offset
        y, x = image.shape[-2], image.shape[-1]
        guassian_mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            guassian_mask[i][landmark[0], landmark[1]] = 1
        
        image = torch.tensor(image)
        image = torch.stack([image, image, image], 0)

        return image, guassian_mask, offset_y, offset_x, landmark_list, item['ID'], scale_rate


class Head_SSL_SAM(Head_Base):
    def __init__(self, pathDataset, mode, size=[512, 512], use_probmap=False, \
        radius_ratio=0.05, min_prob=0.1, tag_ssl=None):
        super().__init__(pathDataset=pathDataset, size=size)

        self.min_prob = min_prob
        self.use_probmap = use_probmap
        self.size = size
        self.patch_size = 400
        # self.patch_size = 256
        self.tag_ssl = tag_ssl


        self.Radius = int(max(size) * radius_ratio)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    guassian_mask[i][j] = math.exp(- 0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.guassian_mask = guassian_mask

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        num_repeat = 10
        if mode == 'Train':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.mode = mode


    def __getitem__(self, index):
        np.random.seed()
        item = self.list[index]

        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
            # print("[debug] ", item['image'].shape)
        
        size = self.size
        
        # # Random translate (crop)
        # size = [self.size[0] - 32, self.size[1] - 32]
        # shift_x = np.random.randint(0, 31)
        # shift_y = np.random.randint(0, 31)
        # item['image'] = item['image'][:, shift_y:shift_y+size[0], shift_x:shift_x+size[1]]
        # probmap = probmap[shift_y:shift_y+size[0], shift_x:shift_x+size[1]]

        # Crop 192 x 192 Patch
        patch_size = self.patch_size

        chosen_x_raw = np.random.randint(int(0.1*size[1]), int(0.9*size[1]))
        chosen_y_raw = np.random.randint(int(0.1*size[0]), int(0.9*size[0]))
        raw_y, raw_x = chosen_y_raw, chosen_x_raw

        left, right = max(0, raw_x - patch_size) + 1, min(size[1] - patch_size, raw_x)
        up, down = max(0, raw_y - patch_size) + 1, min(size[0] - patch_size, raw_y)

        margin_x_1 = np.random.randint(left, right)
        margin_y_1 = np.random.randint(up, down)

        margin_x_2 = np.random.randint(left, right)
        margin_y_2 = np.random.randint(up, down)

        patch_1 = item['image'][:, margin_y_1:margin_y_1+patch_size, margin_x_1:margin_x_1+patch_size]
        patch_2 = item['image'][:, margin_y_2:margin_y_2+patch_size, margin_x_2:margin_x_2+patch_size]

        patch_1 = augment_patch(patch_1, self.aug_transform)
        patch_2 = augment_patch(patch_2, self.aug_transform)

        chosen_y_1, chosen_x_1 = raw_y - margin_y_1, raw_x - margin_x_1
        chosen_y_2, chosen_x_2 = raw_y - margin_y_2, raw_x - margin_x_2

        # to_PIL(patch_1).save('patch_1.jpg')
        # to_PIL(patch_2).save('patch_2.jpg')
        # import ipdb; ipdb.set_trace()

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_1, chosen_x_1] = 1
        temp = cc_augment(torch.cat([patch_1, temp], 0), angle_x=(-np.pi/18, np.pi/18))
        patch_1 = temp[:3]
        temp = temp[3]
        chosen_y_1, chosen_x_1 = temp.argmax() // patch_size, temp.argmax() % patch_size

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_y_2, chosen_x_2] = 1

        temp = cc_augment(torch.cat([patch_2, temp], 0), angle_x=(-np.pi/18, np.pi/18))
        patch_2 = temp[:3]
        temp = temp[3]
        chosen_y_2, chosen_x_2 = temp.argmax() // patch_size, temp.argmax() % patch_size

        rand_global = (torch.rand([500]) * self.patch_size * self.patch_size).int()
        rand_global_y, rand_global_x = rand_global // self.patch_size, rand_global % self.patch_size

        rand_x_min, rand_x_max = max(chosen_x_2.item() - 200, 0), min(chosen_x_2.item() + 200, patch_size)
        rand_y_min, rand_y_max = max(chosen_y_2.item() - 200, 0), min(chosen_y_2.item() + 200, patch_size)
        gap_x, gap_y = rand_x_max - rand_x_min, rand_y_max - rand_y_min
        rand_local = (torch.rand([5000]) * gap_x * gap_y).int()
        rand_local_y, rand_local_x = rand_y_min + rand_local // gap_x, rand_x_min + rand_local % gap_x

        return patch_1, patch_2, [chosen_y_1, chosen_x_1], [chosen_y_2, chosen_x_2], \
            [rand_global_y, rand_global_x], [rand_local_y, rand_local_x]



if __name__ == "__main__":
    # hamming_set(9, 100)
    # test = Test_Cephalometric('../../dataset/Cephalometric', 'Oneshot')
    # for i in range(150):
    #     test.__getitem__(i)
    # test = Head_SSL_Train('../../dataset/Cephalometric', 'Train', use_probmap=True, radius_ratio=0.05, min_prob=0.1)
    test = Head_SSL_SAM('../../dataset/Cephalometric', 'Test')
    for i in range(150):
        test.__getitem__(i)
    print("pass")