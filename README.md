# FM-OSD: Foundation Model-Enabled One-Shot Detection of Anatomical Landmarks (MICCAI 2024)

### Introduction

We provide the codes for FM-OSD on the Head Dataset here.
### Requirements
Please see requirements.txt

### Usage
1. Data preparation:

   Please download the head dataset (Cephalometric) provided by https://github.com/MIRACLE-Center/Oneshot_landmark_detection first and unzip it into the diretory "dataset/Cephalometric/".
   
   Then, please run the data generation code to conduct the offline augmentation. This is applied to the template image and generate 500 augmented images for training.
   ```
   python data_generate.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 41 of the "data_generate.py", the path of the diretory "dataset/Cephalometric/" should be provided. (2) Line 57 of the "data_generate.py", the path of the diretory "data/head" should be provided. Accordingly, Line 13, 14 of the "datasets/head_train.py", the path of the diretory "data/head" should be provided.

2. Train the global branch:

   Please prepare a GPU and run the following code. Our training is conducted on an A40 GPU with 46 GB GPU memory.
   ```
   python train1.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 184, please provide the path of the "output" folder. (2) Line 199, the path of the diretory "dataset/Cephalometric/" should be provided. (3) Line 208, the name of the experiment should be provided. (4) Line 240, the path of the "models" folder should be provided.

3. Train the local branch:

   Please prepare a GPU and run the following code. Our training is conducted on an A40 GPU with 46 GB GPU memory.
   ```
   python train2.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 205, please provide the path of the "output" folder. (2) Line 221, the path of the diretory "dataset/Cephalometric/" should be provided. (3) Line 233, the name of the experiment should be provided. (4) Line 266, the path of the "models" folder should be provided.

4. Test the model:

   We provide the model weights of the global branch and local branch in the "models" folder, denoted as "model_post_iter_9450.pth" and "model_post_fine_iter_20.pth", respectively.
   Please prepare a GPU and run the following code. Our testing is conducted on an A40 GPU with 46 GB GPU memory.
   ```
   python test.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 210, please provide the path of the json file, which records the prediction results. (2) Line 230, please provide the path of the "output" folder. (3) Line 246, the path of the diretory "dataset/Cephalometric/" should be provided. (4) Line 293, please provide the path of the trained model of the local branch after using "python train2.py". (5) Line 299, please provide the path of the trained model of the global branch after using "python train1.py".

### Acknowledgement
This code is based on the framework of [dino-vit-features](https://github.com/ShirAmir/dino-vit-features) and [Oneshot_landmark_detection](https://github.com/MIRACLE-Center/Oneshot_landmark_detection). We thank the authors for their codebase.

## Citation
If you find the code useful for your research, please consider starring ⭐ and cite our paper:
```sh

```

