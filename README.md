# ConViT : Vision Transformers with Convolutional Inductive Biases

This repository contains PyTorch code for ConViT. It is adapted from the open-source implementation of the Data-Efficient Vision Transformer (DeiT), available https://github.com/facebookresearch/deit.

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Training
To train Deit-tiny on ImageNet on a single node with 4 gpus for 300 epochs run:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```

To train ConViT-small in the same conditions, simply add ```--local_up_to_layer 10```, which will exchange the 10 first SA layers for 10 GPSA layers with convolutional initialization.

To train on a subsampled version of ImageNet where we only keep 10% of the training data, use ```--sampling_ratio 0.1```

T


### Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train ConViT-base on ImageNet on 2 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model deit_base_patch16_224 --data-path /path/to/imagenet --local_up_to_layer 10
```

# License
This repository is released under the CC-BY-NC 4.0. license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
