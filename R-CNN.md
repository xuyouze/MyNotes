
# Paper Name:
**_Rich feacture hierarchies for accurate object detection and semantic segmentation_**

# publishing information
R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection
and semantic segmentation. In CVPR, 2014.
[[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)
# 1. background problem:
  * the progress on various visual recognition tasks has been basbeen based on the use of SIFT and HOG, and then the performance has plateaued in last years.
  * there is only a small quantity of annotated detection data.
  * AlexNet show a substantially higher image classification accuracy on the ILSVRC by training a larger CNN on 1.2 million labels images, with ReLU and dropout regularization.
  * nobody use the CNN in object detection.
# 2. the proposed methods:
  * one can apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize and segment object.
  * and when labeled training data is scarce, superivsed pre-training for an auxiliary task, followd by domain-specific fine-tuning, yields a significant performance boost.

# 3. dataset:
  * PASCAL VOC 2010 mAP 53.7%
  * PASCAL VOC 2007 mAP 58.5%
  * ILSVRC 2013 mAP 31.4%
  * PASCAL VOC 2011 segmentation 47.9% 

# 4. advantages:
  * more faster 
    * all CNN parameters are shared across all categories
    * feature vectors computed by the CNN are low-dimensional(4k) when compared to other common approaches.

# 5. the detail of methods:
  * how to localizing objects with a deep network.
    * 

  * how to training a high-capacity model with a small quantity of annotated detection data.



  * first modules: Region proposals there are several approaches as follow
    * ojectness
    * in this work use selective search 
    * category-independent object proposals.
    * constrained parametric min-cuts
    * multi-scale combinatorial grouping 

  * second modules: Feacture extraction
    * regradless of the size or aspect radio of the candidate region, warp all pixels in a tight bounding box around it to the required size, and before warping, dilate the tight bounding box so that at the warped size there are exactly p pixels of warped image context around the original box (use p = 16)
    * use AlexNet as the CNN architecture, extract a 4096-dimensional feature vector from each region proposal.

  * third modules :SVMs

# 6. contribution:
  * this is the first paper to bridge the gap between image classification and object detection and to show that CNN can lead to dramatically higher object detection performance on PASCAL VOC.
  * show that **supervised** pre-training on a large auxiliary dataset(ILSVRC), followed by domain specific fine-tuning on a small dataset(PASCAL), is an effective paradigm for learning high-capacity CNN's when data is scarce

# 7. any questions during the reading :
  * what is selective search:
    answer:
  * what is greedy non-maximum suppression(NMS)?
    answer: for each class independently, rejects a region if it has an intersection-over-union overlap with a higher scoring selected region larger than a learned threshold.
  * previous approach
    * HOG-based deformable part model
    * overfeat
    * UVA detection system

# 8. vocabulary:
plateaued 趋于稳定
canonical 典范
deformable 变形
magnitude 大小
scarce 稀疏
rekindle 重燃
vigorously 大力
distill 提纯
annotate 标注
concurrent 同时
fare well 很好
pedestrian 行人
paradigm 范例
warp 变形
semantic segmentation 语义分割
versus 和
contemporaneous 同期
supression 压制
opt 选择
dilate 膨胀
clobber 破坏
orientation 取向
cortical 皮质
primate 灵长动物
amortize 摊销
resort 采取/度假村
scalable 可扩展性
