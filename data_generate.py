# generate augmented data offline
import argparse
import torch
import numpy as np

from datasets.head import *

from evaluation.eval import *
import os


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

    parser.add_argument('--bs', default=2, type=int, help='batch size.')
    parser.add_argument('--max_iter', default=500, type=int)
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate.')
    parser.add_argument('--exp', default='inner_hard_b2_fast_20231218', help='learning rate.')

    args = parser.parse_args()

    one_shot_loader = Head_SSL_Infer_SSLv1_generate(pathDataset=args.dataset_pth, \
        mode='Oneshot', size=args.input_size, load_size = args.load_size, id_oneshot=args.id_shot)
    
    # data saving path
    snapshot_path = 'xxx/data/head/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    image_root = snapshot_path + 'image/'
    label_root = snapshot_path + 'label/'

    if not os.path.exists(image_root):
        os.makedirs(image_root)
    if not os.path.exists(label_root):
        os.makedirs(label_root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_performance = 10000
    
    for iter_num in range(args.max_iter):
        image, landmarks_temp, _ = one_shot_loader.__getitem__(0)

        image_path = image_root + str(args.id_shot) + '_{}.png'.format(iter_num)
        label_path = label_root + str(args.id_shot) + '_{}.npy'.format(iter_num)

        image.save(image_path)
        np.save(label_path, landmarks_temp)
        