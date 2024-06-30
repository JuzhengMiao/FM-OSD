# testing code
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
from tqdm import tqdm
import numpy as np

from datasets.head_train import *
from datasets.head import *
from torch.utils.data import DataLoader
import ipdb
from evaluation.eval import *
from post_net import *
import torch.nn as nn

import torch.optim as optim
import os

def find_landmark_all(extractor, device, model_post, image_path1: str, dataloader, lab, load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                         stride: int = 4, original_size = [2400, 1935], topk = 5):
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size

    descriptors1_post = model_post(descriptors1, num_patches1, load_size1, islocal = False)  
    descriptors1_post_large = torch.nn.functional.interpolate(descriptors1_post, original_size, mode = 'bilinear')

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

        image1_batch_local, _, gt_local, offset, crop_feature = extractor.preprocess_local_withfeature(image_path1, load_size, [int(lab[i][0]), int(lab[i][1])], descriptors1_post_large)
        gt_local_all.append(gt_local)
        descriptors1_local = extractor.extract_descriptors(image1_batch_local.to(device), layer, facet, bin)
        num_patches1_local, load_size1_local = extractor.num_patches, extractor.load_size  
        descriptors1_post_local = model_post(descriptors1_local, num_patches1_local, load_size1_local, islocal = True)  
        
        descriptors1_post_local = nn.functional.normalize(descriptors1_post_local, dim=1) + nn.functional.normalize(crop_feature, dim=1)

        descriptors1_post_local_all.append(descriptors1_post_local)
        lab_feature_local = descriptors1_post_local[0, :, gt_local[0], gt_local[1]]
        lab_feature_local = lab_feature_local.unsqueeze(1).unsqueeze(2)
        lab_feature_all_local.append(lab_feature_local)

    final_dict = {}
    final_dict["images"] = []
    local_dict = {}
    local_dict['name'] = image_path1
    local_dict['gt'] = lab

    final_dict["template"] = local_dict

    pred_all = []
    gt_all = []

    imgs_all = []
    img_names_all = []

    # iterate over all the testing images
    for image, landmark_list, img_path_query in tqdm(dataloader):
        local_dict = {}
        local_dict['name'] = img_path_query[0]
        
        image2_batch, image2_pil = extractor.preprocess(img_path_query[0], load_size)
        descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
        num_patches2, load_size2 = extractor.num_patches, extractor.load_size

        descriptors2_post = model_post(descriptors2, num_patches2, load_size2, islocal = False)
        descriptors2_post_large = torch.nn.functional.interpolate(descriptors2_post, original_size, mode = 'bilinear')

        size_y, size_x = descriptors1_post.shape[-2:]

        points1 = []
        points2 = []

        points2_c = []

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

            points2_c.append([y2_show, x2_show])


            distance1 = (y2_show - landmark_list[i][0].item())*(y2_show - landmark_list[i][0].item()) + (x2_show - landmark_list[i][1].item())*(x2_show - landmark_list[i][1].item())
            distance1 = np.sqrt(distance1)

            image2_batch_local, _, gt_local2, offset2, crop_feature2 = extractor.preprocess_local_withfeature(img_path_query[0], load_size, [int(y2_show), int(x2_show)], descriptors2_post_large)
            descriptors2_local = extractor.extract_descriptors(image2_batch_local.to(device), layer, facet, bin)
            num_patches2_local, load_size2_local = extractor.num_patches, extractor.load_size

            descriptors2_post_local = model_post(descriptors2_local, num_patches2_local, load_size2_local, islocal = True)

            descriptors2_post_local = nn.functional.normalize(descriptors2_post_local, dim = 1) + nn.functional.normalize(crop_feature2, dim = 1)
            
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


            distance2 = (y2_show - landmark_list[i][0].item())*(y2_show - landmark_list[i][0].item()) + (x2_show - landmark_list[i][1].item())*(x2_show - landmark_list[i][1].item())
            distance2 = np.sqrt(distance2)

            points2.append([y2_show, x2_show])

        pred_all.append(points2)
        gt_all.append(points1)

        local_dict['gt'] = points1
        local_dict['pred'] = points2  
        local_dict['pred_c'] = points2_c
        final_dict["images"].append(local_dict)


    with open("xxx/results/ours_head_topk3.json", "w") as f:
        json.dump(final_dict, f)
    print('success saving')
    return pred_all, gt_all, imgs_all, img_names_all  # lists



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--save_dir', type=str, default = 'xxx/output', required=False, help='The root save dir for results.')
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
    parser.add_argument('--topk', default=3, type=int, help='Final number of correspondences.')

    # 
    parser.add_argument('--dataset_pth', type=str, default = 'xxx/dataset/Cephalometric/', required=False, help='data root')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int, help='template id')
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8], help='radius')

    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='direct_up_mse_20231226', help='learning rate.')

    args = parser.parse_args()

    # fix random seed
    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # define one_shot loader
    train_dataset = TrainDataset(istrain = 0, original_size = args.input_size, load_size = args.load_size)
    
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    best_performance = 10000
    
    iter_num = 0
    # load models
    model_path = 'xxx/models/model_post_fine_iter_20.pth'
    model_post.load_state_dict(torch.load(model_path))

    # load global model
    model_dict = model_post.state_dict()

    global_model_path = 'xxx/models/model_post_iter_9450.pth'
    pretrained_dict = torch.load(global_model_path)
    
    model_keys = model_dict.keys()
    keys = [k for k in model_keys if 'conv_out1' in k]

    values = []
    for k in keys:
        temp = k.split('.')
        key_temp = temp[0][:-1]
        for i in range(len(temp) - 1):
            key_temp += '.' + temp[i+1]
        values.append(pretrained_dict[key_temp])
        
    new_state_dict = {k: v for k, v in zip(keys, values)}
    model_dict.update(new_state_dict)
    model_post.load_state_dict(model_dict)
    model_post.eval()
    with torch.no_grad():
        pred_all, gt_all, imgs_all, img_names_all = find_landmark_all(extractor, device, model_post, img_path_temp, dataloader_test, landmarks_temp_val, args.load_size, args.layer, args.facet, args.bin, args.thresh, args.model_type, args.stride, topk = args.topk)
       
    test_name = 'dino_s'
    save_root = args.save_dir + '/' + test_name
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    evaluater = Evaluater(pred_all, gt_all, args.eval_radius, save_root, name = 'all', spacing = [0.1, 0.1], imgs = imgs_all, img_names = img_names_all)
    evaluater.calculate()
    evaluater.cal_metrics()