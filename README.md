# [DiSK_Distilling_Scaffolded_Knowledge](https://openreview.net/forum?id=N4K5ck-BTT)

We develop a novel KD scheme where the teacher scaffolds the student's prediction on hard-to-learn examples.  It smoothens student's loss landscape so that the student encounters fewer local minima. As a result it has good generalization properties.

## Brief Description  

![](<pdf/PDE-GlobalLayer-Poster.png>)


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

```
bash runner.sh
```

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
