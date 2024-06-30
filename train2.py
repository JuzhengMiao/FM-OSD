# train the local branch
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple

from datasets.head_train import *
from datasets.head import *
from torch.utils.data import DataLoader
# import ipdb
from evaluation.eval import *
from post_net import *
import torch.nn as nn

import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from PIL import ImageDraw, ImageFont
import cv2
import time


def find_landmark_all_local(extractor, device, model_post, image_path1: str, dataloader, lab, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4, original_size = [2400, 1935], topk = 5):
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size

    descriptors1_post = model_post(descriptors1, num_patches1, load_size1, islocal = False)  

    # obtain the template features
    lab_feature_all = []
    descriptors1_post_local_all = []
    lab_feature_all_local = []

    gt_local_all = []

    for i in range(len(lab)):
        lab_y = int(lab[i][0]) 
        lab_x = int(lab[i][1])
        size_y, size_x = descriptors1_post.shape[-2:]
        lab_y = int(lab_y/original_size[0]*size_y)   
        lab_x = int(lab_x/original_size[1]*size_x)

        lab_feature = descriptors1_post[0, :, lab_y, lab_x]

        lab_feature = lab_feature.unsqueeze(1).unsqueeze(2)
        lab_feature_all.append(lab_feature)

        image1_batch_local, _, gt_local, offset = extractor.preprocess_local(image_path1, load_size, [int(lab[i][0]), int(lab[i][1])])
        gt_local_all.append(gt_local)
        descriptors1_local = extractor.extract_descriptors(image1_batch_local.to(device), layer, facet, bin)
        num_patches1_local, load_size1_local = extractor.num_patches, extractor.load_size 
        descriptors1_post_local = model_post(descriptors1_local, num_patches1_local, load_size1_local, islocal = True) 
        descriptors1_post_local_all.append(descriptors1_post_local)
        lab_feature_local = descriptors1_post_local[0, :, gt_local[0], gt_local[1]]
        lab_feature_local = lab_feature_local.unsqueeze(1).unsqueeze(2)
        lab_feature_all_local.append(lab_feature_local)

    pred_all = []
    gt_all = []

    imgs_all = []
    img_names_all = []

    # iterate over all the testing images
    for image, landmark_list, img_path_query in tqdm(dataloader):
        image2_batch, image2_pil = extractor.preprocess(img_path_query[0], load_size)
    
        points1 = []
        points2 = []

        imgs_all.append(image)
        img_names_all.append(img_path_query)

        # iterate over each point
        for i in range(len(lab)):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])

            y2_show = np.round(landmark_list[i][0].item())
            x2_show = np.round(landmark_list[i][1].item())
            
            image2_batch_local, _, gt_local2, offset2 = extractor.preprocess_local(img_path_query[0], load_size, [int(y2_show), int(x2_show)])
            descriptors2_local = extractor.extract_descriptors(image2_batch_local.to(device), layer, facet, bin)
            num_patches2_local, load_size2_local = extractor.num_patches, extractor.load_size

            descriptors2_post_local = model_post(descriptors2_local, num_patches2_local, load_size2_local, islocal = True)

            similarities_local = torch.nn.CosineSimilarity(dim=0)(lab_feature_all_local[i], descriptors2_post_local[0])

            h2, w2 = similarities_local.shape
            similarities_local = similarities_local.reshape(1, -1).squeeze(0) 
            sim_k_local, nn_k_local = torch.topk(similarities_local, k = topk, dim=-1, largest=True)

            distance_best_local = 1000000000
            index_best_local = 0
            for index_local in range(topk):
                i_y = nn_k_local[index_local]//w2
                i_x = nn_k_local[index_local]%w2
                similarities_reverse_local = torch.nn.CosineSimilarity(dim=0)(descriptors2_post_local[0, :, i_y, i_x].unsqueeze(1).unsqueeze(2), descriptors1_post_local_all[i][0])
                h1, w1 = similarities_reverse_local.shape
                similarities_reverse_local = similarities_reverse_local.reshape(-1) 
                _, nn_1_local = torch.max(similarities_reverse_local, dim=-1)
                img1_y_to_show_local = nn_1_local // w1
                img1_x_to_show_local = nn_1_local % w1

                size_y, size_x = descriptors1_post_local_all[i].shape[-2:]
            
                x1_show = img1_x_to_show_local
                y1_show = img1_y_to_show_local
                    
                distance_temp_local = pow(y1_show - gt_local_all[i][0], 2) + pow(x1_show - gt_local_all[i][1], 2)
                if distance_temp_local < distance_best_local:
                    distance_best_local = distance_temp_local
                    index_best_local = index_local
            
            img2_indices_to_show_local = nn_k_local[index_best_local:index_best_local+1].cpu().item()

            size_y, size_x = descriptors2_post_local.shape[-2:]
            y2_show_local = img2_indices_to_show_local // size_x
            x2_show_local = img2_indices_to_show_local % size_x

            y2_show = offset2[0] + y2_show_local
            x2_show = offset2[1] + x2_show_local

            points2.append([y2_show, x2_show])

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, imgs_all, img_names_all # lists

def get_feature(extractor, device, image1_batch, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4, original_size = [2400, 1935], topk = 5):
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size

    return descriptors1, num_patches1, load_size1

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_heatmap(landmark, size, var=5.0):   
    length, width = size
    x,y = torch.meshgrid(torch.arange(0, length),
                         torch.arange(0,width))
    p = torch.stack([x,y], dim=2).float()
    inner_factor = -1/(2*(var**2))
    mean = torch.as_tensor(landmark).float()
    heatmap = (p-mean).pow(2).sum(dim=-1)
    heatmap= torch.exp(heatmap*inner_factor)
    return heatmap

def heatmap_mse_loss(features, landmarks, var = 5.0, criterion = torch.nn.MSELoss()):
    lab = []
    for i in range(len(landmarks)): 
        labels = landmarks[i]
        labtemp = []
        for l in range(labels.shape[0]):
            labtemp.append(make_heatmap(labels[l], [features.shape[-2], features.shape[-1]], var=var))

        labtemp2 = torch.stack(labtemp, dim = 0)
        lab.append(labtemp2)

    label = torch.stack(lab, dim = 0)
    label = label.to(features.device)

    pred = []
    for i in range(len(landmarks)):  # batchsize
        feature_temp = features[i]  
        pred_temp = []
        for j in range(labels.shape[0]): # number of landmarks
            gt = feature_temp[:, landmarks[i,j,0], landmarks[i,j,1]].unsqueeze(1).unsqueeze(2)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0)  # 1*h*w
            pred_temp.append(similarity)

        pred_temp = torch.cat(pred_temp, dim = 0).unsqueeze(0)
        pred.append(pred_temp)
    
    pred = torch.cat(pred, dim = 0)

    loss = criterion(pred, label)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--save_dir', type=str, default = 'xxx/output', required=False)
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=8, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='True', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.05, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--topk', default=5, type=int, help='Final number of correspondences.')

    # 
    parser.add_argument('--dataset_pth', type=str, default = 'xxx/dataset/Cephalometric/', required=False, help='data path')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int, help='template id')
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8], help='radius')


    parser.add_argument('--bs', default=4, type=int, help='batch size.')
    parser.add_argument('--random_range', default=50, type=int, help='random_range.')
    parser.add_argument('--local_coe', default=1.0, type=float, help='local_coe.')
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='local', help='learning rate.')

    args = parser.parse_args()

    # random seed
    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    train_dataset = TrainDataset(istrain = 0, original_size = args.input_size, load_size = args.load_size)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    
    one_shot_loader_val = Head_SSL_Infer(pathDataset=args.dataset_pth, \
        mode='Oneshot', size=args.input_size, id_oneshot=args.id_shot)
    
    _, landmarks_temp_val, img_path_temp = one_shot_loader_val.__getitem__(0)
    
    
    dataset_test = Head_SSL_Infer(args.dataset_pth, mode = 'Test', size=args.input_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                shuffle=False, num_workers=2)
    
    image, _, _, _ = train_dataset.__getitem__(0)

    image_size = (image.shape[-2], image.shape[-1])
    model_post = Upnet_v3_coarsetofine2_tran_new(image_size, 6528, 256).cuda()
    model_post.train()

    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)

    # model saving path
    snapshot_path = 'xxx/models/' + args.exp
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    writer = SummaryWriter(snapshot_path + '/log')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    best_performance = 10000
    
    iter_num = 0
    max_iterations = args.max_iterations
    # testing at first
    model_post.eval()
    with torch.no_grad():
        pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_local(extractor, device, model_post, img_path_temp, dataloader_test, landmarks_temp_val, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)
    print('prediction finished')
    test_name = 'dino_s'
    save_root = args.save_dir + '/' + test_name
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    evaluater = Evaluater(pred_all, gt_all, args.eval_radius, save_root, name = 'stride4_224_mse_key_layer_' + str(args.layer) + '_' + str(args.id_shot), spacing = [0.1, 0.1], imgs = imgs_all, img_names = img_names_all)
    evaluater.calculate()
    evaluater.cal_metrics()

    performance = evaluater.mre
    
    save_best = os.path.join(snapshot_path, 'model_post_fine_iter_{}_{}.pth'.format(iter_num, performance))
    torch.save(model_post.state_dict(), save_best)
    model_post.train()
    for epoch in np.arange(0, args.max_epoch) + 1:
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_dataloaders:
            iter_num = iter_num + 1
            optimizer.zero_grad()
            for j in range(labs.shape[1]): 
                image_local = []
                lab_local = []
                for i in range(images.shape[0]):
                    image_local_j, _, gt_local2, _ = extractor.preprocess_local_random_new(image_paths[i], args.load_size, [int(labs[i,j,0]), int(labs[i,j,1])], args.random_range)
                    image_local.append(image_local_j)
                    lab_local.append(gt_local2)

                image_local = torch.cat(image_local, dim = 0)
                lab_local = torch.tensor(lab_local).unsqueeze(1)
                with torch.no_grad():
                    descriptors_local, num_patches_local, load_size_local = get_feature(extractor, device, image_local, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)
                descriptors_post_local = model_post(descriptors_local, num_patches_local, load_size_local, islocal = True) 

                loss_local = args.local_coe*heatmap_mse_loss(descriptors_post_local, lab_local, var = 2.0)/labs.shape[1]

                loss_local.backward(retain_graph=True)
                
                writer.add_scalar('info/loss_local_{}'.format(j), loss_local, iter_num)
            optimizer.step()
            print('iter: {}, loss_local: {}'.format(iter_num, loss_local))

            # regular testing and saving
            if iter_num % 20 == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_local(extractor, device, model_post, img_path_temp, dataloader_test, landmarks_temp_val, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)
                test_name = 'dino_s'
                save_root = args.save_dir + '/' + test_name
                save_root = Path(save_root)
                save_root.mkdir(exist_ok=True, parents=True)
                evaluater = Evaluater(pred_all, gt_all, args.eval_radius, save_root, name = 'stride4_224_mse_key_layer_' + str(args.layer) + '_' + str(args.id_shot), spacing = [0.1, 0.1], imgs = imgs_all, img_names = img_names_all)
                evaluater.calculate()
                evaluater.cal_metrics()

                performance = evaluater.mre
                writer.add_scalar('info/performance', performance, iter_num)
                if performance < best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path, 'model_post_fine_iter_{}_{}.pth'.format(iter_num, best_performance))
                    torch.save(model_post.state_dict(), save_best)
                model_post.train()
            
            if iter_num % 100 == 0:
                # regular saving
                save_path = os.path.join(snapshot_path, 'model_post_fine_iter_{}.pth'.format(iter_num))
                torch.save(model_post.state_dict(), save_path)
            
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
                break

