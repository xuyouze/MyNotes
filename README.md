# This is my notebook about papers in what I have read.
paper list as follow:
- [x] **_ImageNet Classification with Deep Convolutional Neural Networks_** [AlexNet.md](./AlexNet.md)
- [x] **_Multi-task learning of cascaded CNN for facial attribute classification PR_**  [FAC.md](./FAC.md)
- [x] **_Network In Networks_**  [NIN.md](./NIN.md)
- [x] **_VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION_**  [VGG.md](./VGG.md)
- [x] **_Going deeper with convolution_** [GoogLeNet.md](./GoogLeNet.md)
- [x] **_Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_** [BatchNormalization.md](./BatchNormalization.md)
- [x] **_Rich feacture hierarchies for accurate object detection and semantic segmentation_**  [R-CNN.md](./R-CNN.md)
- [x] **_Deep Residual Learning for Image Recognition_** [ResNet.md](./ResNet.md)

- [ ] **_Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification_** [DelvingDeepintoRectifiers.md](./DelvingDeepIntoRectifiers.md)

论文的缺陷:
  * 速度太慢
  *  精度太低
  *  应用场景太简单
  *  泛化能力差
  *  对某些特定问题处理不够好

应对的大体思路:
  * 样本太小导致的问题，对数据进行增强
  *  对网络组件进行增、删、改
	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%