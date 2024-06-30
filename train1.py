# train the global branch
import argparse
import torch
from pathlib import Path
# from extractor import ViTExtractor
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
import ipdb
from evaluation.eval import *
from post_net import *
import torch.nn as nn

import torch.optim as optim
from tensorboardX import SummaryWriter
import os

def find_landmark_all(extractor, device, model_post, image_path1: str, dataloader, lab, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4, original_size = [2400, 1935], topk = 5):
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size

    descriptors1_post = model_post(descriptors1, num_patches1)

    # obtain the template features
    lab_feature_all = []

    for i in range(len(lab)):
        lab_y = int(lab[i][0])
        lab_x = int(lab[i][1])
        size_y, size_x = descriptors1_post.shape[-2:]
        lab_y = int(lab_y/original_size[0]*size_y)  
        lab_x = int(lab_x/original_size[1]*size_x)

        lab_feature = descriptors1_post[0, :, lab_y, lab_x]

        lab_feature = lab_feature.unsqueeze(1).unsqueeze(2)
        lab_feature_all.append(lab_feature)

    pred_all = []
    gt_all = []

    imgs_all = []
    img_names_all = []

    # iterate over all the testing images
    for image, landmark_list, img_path_query in tqdm(dataloader):
        image2_batch, image2_pil = extractor.preprocess(img_path_query[0], load_size)
        descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
        num_patches2, load_size2 = extractor.num_patches, extractor.load_size

        descriptors2_post = model_post(descriptors2, num_patches2)

        points1 = []
        points2 = []

        imgs_all.append(image)
        img_names_all.append(img_path_query)

        # iterate over each point
        for i in range(len(lab)):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])

            # compute similarity
            similarities = torch.nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])  

            h2, w2 = similarities.shape
            similarities = similarities.reshape(1, -1).squeeze(0)  
            sim_k, nn_k = torch.topk(similarities, k = topk, dim=-1, largest=True)

            # inverse direction
            distance_best = 1000000000
            index_best = 0
            for index in range(topk):
                i_y = nn_k[index]//w2
                i_x = nn_k[index]%w2
                similarities_reverse = torch.nn.CosineSimilarity(dim=0)(descriptors2_post[0, :, i_y, i_x].unsqueeze(1).unsqueeze(2), descriptors1_post[0])
                h1, w1 = similarities_reverse.shape
                similarities_reverse = similarities_reverse.reshape(-1) 
                _, nn_1 = torch.max(similarities_reverse, dim=-1)
                img1_y_to_show = nn_1 // w1
                img1_x_to_show = nn_1 % w1

                size_y, size_x = descriptors1_post.shape[-2:]
            
                x1_show = img1_x_to_show/size_x * original_size[1]
                y1_show = img1_y_to_show/size_y * original_size[0]

                distance_temp = pow(y1_show - int(lab[i][0]), 2) + pow(x1_show - int(lab[i][1]),2)
                if distance_temp < distance_best:
                    distance_best = distance_temp
                    index_best = index

            img2_indices_to_show = nn_k[index_best:index_best+1].cpu().item()
           
            size_y, size_x = descriptors2_post.shape[-2:]
            y2_show = img2_indices_to_show // size_x
            x2_show = img2_indices_to_show % size_x

            y2_show = np.round(y2_show/size_y*original_size[0])
            x2_show = np.round(x2_show/size_x*original_size[1])
            points2.append([y2_show, x2_show])

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, imgs_all, img_names_all  # lists

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
    for i in range(len(landmarks)):     # batchsize
        labels = landmarks[i]
        labtemp = []
        for l in range(labels.shape[0]):
            labtemp.append(make_heatmap(labels[l], [features.shape[-2], features.shape[-1]], var=var))
        labtemp2 = torch.stack(labtemp, dim = 0)
        lab.append(labtemp2)

    label = torch.stack(lab,dim=0)
    label = label.to(features.device)

    pred = []
    for i in range(len(landmarks)):  # batchsize
        feature_temp = features[i]
        pred_temp = []
        for j in range(labels.shape[0]): # number of landmarks
            gt = feature_temp[:, landmarks[i,j,0], landmarks[i,j,1]].unsqueeze(1).unsqueeze(2)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0) 
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

    parser.add_argument('--dataset_pth', type=str, default = 'xxx/dataset/Cephalometric/', required=False, help='data path')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int, help='template id')
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8], help='radius')

    parser.add_argument('--bs', default=4, type=int, help='batch size.')
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=20000)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='global', help='exp name.')

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
    model_post = Upnet_v3(image_size, 6528, 256).cuda()
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
        pred_all, gt_all, imgs_all, img_names_all = find_landmark_all(extractor, device, model_post, img_path_temp, dataloader_test, landmarks_temp_val, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)

    test_name = 'dino_s'
    save_root = args.save_dir + '/' + test_name
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    evaluater = Evaluater(pred_all, gt_all, args.eval_radius, save_root, name = 'stride4_224_mse_key_layer_' + str(args.layer) + '_' + str(args.id_shot), spacing = [0.1, 0.1], imgs = imgs_all, img_names = img_names_all)
    evaluater.calculate()
    evaluater.cal_metrics()

    performance = evaluater.mre

    save_best = os.path.join(snapshot_path, 'model_post_iter_{}_{}.pth'.format(iter_num, performance))
    torch.save(model_post.state_dict(), save_best)
    model_post.train()
    for epoch in np.arange(0, args.max_epoch) + 1:
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_dataloaders:
            iter_num = iter_num + 1
            with torch.no_grad():
                descriptors, num_patches, load_size = get_feature(extractor, device, images, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)

            descriptors_post = model_post(descriptors, num_patches)  
        
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)
            loss.backward()
            optimizer.step()

            writer.add_scalar('info/loss', loss, iter_num)
            print('iter: {}, loss: {}'.format(iter_num, loss))

            # regular testing and saving
            if iter_num % 50 == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all(extractor, device, model_post, img_path_temp, dataloader_test, landmarks_temp_val, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)

                test_name = 'dino_s'
                save_root = args.save_dir + '/' + test_name
                save_root = Path(save_root)
                save_root.mkdir(exist_ok=True, parents=True)
                evaluater = Evaluater(pred_all, gt_all, args.eval_radius, save_root, name = 'stride4_224_mse_key_layer_' + str(args.layer) + '_' + str(args.id_shot), spacing = [0.1, 0.1], imgs = imgs_all, img_names = img_names_all)
                evaluater.calculate()
                evaluater.cal_metrics()

                performance = evaluater.mre
                if performance < best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path, 'model_post_iter_{}_{}.pth'.format(iter_num, best_performance))
                    torch.save(model_post.state_dict(), save_best)
                model_post.train()
            
            if iter_num % 3000 == 0:
                # regular saving
                save_path = os.path.join(snapshot_path, 'model_post_iter_{}.pth'.format(iter_num))
                torch.save(model_post.state_dict(), save_path)
            
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
                break
