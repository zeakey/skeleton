# Object Skeleton Detection with FCN
## How to use
### For training:
1. clone source and build caffe inside;
2. download [SK-LARGE](http://kaiz.xyz/sk-large) dataset to `skeleton/data/` and do data augmentation follow instructions there;
3. download initial model [vgg16convs.caffemodel](http://zhaok-data.oss-cn-shanghai.aliyuncs.com/caffe-model/vgg16convs.caffemodel) to `skeleton/models/`;
4. `python train.py`.

Refer to [project page](http://kaiz.xyz/deepsk) for more instruction on how to use this code.
___
KAI ZHAO  <http://kaiz.xyz>
