# Models
## Corresponding papers:
* fsds: *Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs*
* h1/h2: *Hi-Fi: Hierarchical Feature Integration for Skeleton Detection*

## Note that:
1. `h1` represents 1 leverl hierarchical feature integration and `h2` represents 2 level integration, refer to the
  [paper](http://data.kaiz.xyz/publications/zhao2018hifi.pdf) for details;
2. Both FSDS and Hi-Fi (h1 and h2) requires quantized skeleton scale maps as supervision*;

## For these who want to try Hi-Fi but do not have scale annotations:
We provide other Hi-Fi architectures which are suitable for binary tasks (such as edge detection):

* <https://github.com/zeakey/hed/blob/master/model/h1.py>
* <https://github.com/zeakey/hed/blob/master/model/h2.py>

## Citation:
Cite papers below if you use the model(s) in you research:

* FSDS:
```
@article{shen2016object,
  title={Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs},
  author={Shen, Wei and Zhao, Kai and Jiang, Yuan and Wang, Yan and Zhang, Zhijiang and Bai, Xiang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016},
  pages={222-230},
  publisher={IEEE},
}
```

* Hi-Fi:
```
@article{zhao2018hifi,
  title={Hi-Fi: Hierarchical Feature Integration for Skeleton Detection},
  author={Zhao, Kai and Shen, Wei and Gao, Shanghua and Li, Dandan and Cheng, Ming-Ming},
  journal={Preceding of the International Joint Conference on Artificial Intelligence, 2018},
  year={2018}
}
```
___
<http://kaiz.xyz>
