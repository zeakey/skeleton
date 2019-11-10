# Object Skeleton Detection with FCN
This repo contains codes for our relevant papers on **Object Skeleton Detection in Natural Images**, they are

1. Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs, **CVPR2016**
2. DeepSkeleton: Learning Multi-task Scale-associated Deep Side Outputs for Object Skeleton Extraction in Natural Images, **IEEE Trans on Image Processing**
3. Hi-Fi: Hierarchical Feature Integration for Skeleton Detection, **IJCAI2018**


### Simple Steps For training:
1. clone source and build caffe inside;
2. download [SK-LARGE](http://kaiz.xyz/sk-large) dataset to `skeleton/data/` and do data augmentation follow instructions there;
3. download initial model <http://data.kaizhao.net/projects/edge/vgg16convs.caffemodel> to `skeleton/models/`;
4. `python train.py` (by default we use the 'FSDS' which is proposed in paper[1]).

Refer to <http://kaizhao.net/deepsk> for detailed instructions, frequently asked questions and pretrained models.


### FAQ
[Frequently asked questions](FAQ.md)
___

*Copyright* [KAI ZHAO](http://kaiz.xyz)
