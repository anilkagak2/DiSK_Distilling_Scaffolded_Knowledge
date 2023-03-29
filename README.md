# [DiSK: Distilling Scaffolded Knowledge aka Scaffolding a Student to Instill Knowledge](https://openreview.net/forum?id=N4K5ck-BTT)

We develop a novel KD scheme where the teacher scaffolds the student's prediction on hard-to-learn examples.  It smoothens student's loss landscape so that the student encounters fewer local minima. As a result it has good generalization properties.

## Brief Description  

![](<pdf/PDE-GlobalLayer-Poster.png>)

### ImageNet Results

| **Student**               | **Teacher**               | **Cross-Entropy** | **Existing  Distillation Literature** | **DiSK** |
|---------------------------|---------------------------|-------------------|---------------------------------------|----------|
| ResNet18                  | ResNet50                  | [69.73](https://github.com/huggingface/pytorch-image-models/blob/v0.5.4/results/results-imagenet.csv)             | 71.29                                 | 72.35    |
| ViT-Tiny (Patch 16, 224)  | ViT-Large (Patch 16, 384) | [75.45](https://github.com/huggingface/pytorch-image-models/blob/v0.5.4/results/results-imagenet.csv)             |                                       | [77.86](https://drive.google.com/file/d/1bT05AYapjbjpkiJQoxJFBLLkHZYN8X_H/view?usp=share_link)    |
| DeiT-Tiny (Patch 16, 224) | ViT-Large (Patch 16, 384) | [72.2](https://github.com/facebookresearch/deit/blob/main/README_deit.md)              | [74.5](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                                  | [75.59](https://drive.google.com/file/d/1bPYCATBjql-AlxZ4MGbb5tU_M1kmz63H/view?usp=share_link)    |


## Installation

Our codebase is written using [PyTorch](https://pytorch.org). You can set up the environment using [Conda](https://www.anaconda.com/products/individual) and executing the following commands.  

```
conda create --name pytorch-1.10 python=3.9
conda activate pytorch-1.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Please update the last command as per your system specifications (see [PyTorch-Install](https://pytorch.org/get-started/locally/)). Although we have not explicitly tested all the recent PyTorch versions, but you should be able to run this code on PyTorch>=1.7 and Python>=3.7


Please install the following packages used in our codebase.

```
pip install tqdm
pip install thop
pip install timm==0.5.4
pip install pyyaml
```

## Training Scripts 

Our DiSK.py file contains the core utilities including the proposed scaffolded distillation logic. You can run the CIFAR-100 and Tiny-ImageNet experiments using the runner.sh file by updating the student and teacher networks as well as their pre-trained models. For ImageNet experiments, follow the runner.sh file in the imagenet folder. 

We provide access to the cross-entropy pre-trained teacher and student models in the [google drive link](https://drive.google.com/drive/folders/1ZDJTXiPAzKOMd8n9wsHCS2-PQWHTbouS?usp=share_link). Please download the required models in your local directory and provide this directory location in the ```runner.sh``` script.

```
bash runner.sh
```

In order to train cross-entropy based teacher/student models, simply update the model name in the ```runner_ce.sh``` script and run 
```
bash runner_ce.sh
```

Note that our scripts support CIFAR-100 and Tiny-ImageNet datasets out of the box. ImageNet experiments are separated in the sub-directory as they rely on ```timm``` repository for providing definitions of wide-range of vision architectures as well as distributed training on multiple gpus. 

You can extend this code to your custom dataset by hacking various definitions in the dataset and model definition folders, and adding access points in the model dictionary.

## Reference (Bibtex entry)


```
@inproceedings{kag2023scaffolding,
  title     = {Scaffolding a Student to Instill Knowledge},
  author    = {Anil Kag and Durmus Alp Emre Acar and Aditya Gangrade and Venkatesh Saligrama},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=N4K5ck-BTT}
}
```
