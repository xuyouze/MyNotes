
# Paper Name:
Deep Learning Face Attribute in the Wild

# publishing information
Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: International Conference on Computer Vision, IEEE (2015) 3730–3738

# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:
subtle 微小的

# 中文翻译
## abstract
面部属性识别很难，提出了一个学习框架，由两个级联网络组成，LNet、ANet，分别用不同的数据集预训练过，然后用CelebA fine-tuned，
效果很好，也揭示了很多的事实：
定位跟预测可以通过不同的数据进行预训练，
揭示了尽管LNet的过滤器只用图片层次的属性标签进行优化，但是有很强的脸部位置指示。这也展示了高层次的ANet隐藏单元在经过大量的预训练之后能够自动发现语义信息，在使用了属性标签进行优化之后能够被大大增强
## introduction
脸部属性识别有很多的应用
现在属性识别方法分为两种，全局跟局部方法，全局方法不够好，不鲁棒，局部的也不好，比如HOG跟SVM组合起来的方法。
我的工作算全局方法，两个级联网络，创新点在三个地方：
LNet是用弱监督方法训练，用图像级别的属性标签的图片来训练，不需要脸部的box，使用了大量通用目标类别来预训练，实验说明预训练很有效，
ANet 可以提取不同的特征，让属性识别成为可能，使用大量脸部图片进行预训练，然后用属性进行优化，ANet能够解释在无约束脸部图片的复杂变化。
在LNet 提供的脸部区域的大概位置下，平均多个patch的预测可以提高效果，本地共享过滤器在面部相关任务中表现更好。 这通过提出交织操作来解决。
提出了许多在学习脸部表示的有价值的观点，
分别使用大量目标类别的图片、脸部图片可以提高定位、识别属性的的效果。
主要贡献：
效果有很大的进步
设计了一种具有局部共享滤波器的CNN快速前馈算法。
提供了CelebA数据集
## related work
介绍了之前的相关做法，比如 HOG+SVM、SVM、CNN，这些方法都有一些缺点，训练的时候需要脸部标志标注数据集，严重依赖于准确的特征点检测，跟姿态识别。
## approach
LNet使用ImageNet预训练，然后用图片层级的属性标签进行优化，预训练是解释背景杂乱，优化是为了学习对复杂脸部变换的鲁棒性特征。
LNet 结构跟AlexNet很像，LNet$_o$ 的C5 代表头肩，LNet$_s$代表头部
LNet 第一个输入为 m*n,输出为3\*3\*256,取出平均响应图中具有高响应区域，resize 成227\*227的图片，作为肩头部，以及第二个LNet的输入，输出还是3\*3\*256，然后取出平均响应图中具有高响应区域，作为头部作为 ANet 的输入，以及SVM的输入，
ANet包含四个卷积层，其中C1和C2的滤波器是全局共享的，C3和C4的滤波器是本地共享的
### face localization
通过预训练，LNet可以正确定位人脸
优化的时候，两个LNet都增加一个输出层，训练两次，测试的时候移除，
LNet0 是拿整张图输入，LNets是拿训练好的LNet0的响应图进行训练，使用多个sigmoid激活函数作为最后一层，然后累加作为loss函数
经过测试发现没有预训练效果很差

通过2k张图片的最大值的直方图，判断阈值，越大的阈值越有可能是人脸，
实验证明，使用多个属性进行训练会有更好的定位效果。

### attribute prediction
ANet 拿来提取特征，SVM对特征进行分类。
ANet先用脸部例子预训练，然后拓展定位的脸部区域，是为了扩展更多的信息，然后裁剪多个patch，作为ANet的输入，来学习高层次的特征。
最后使用featrue作为svm的输入进行预测。
预训练，使用160k的celebAFace图片，使用了结合softmax跟similarity 的loss
提出了一个可以减少卷积操作的方法，

## 疑惑
啥是 global share、 locally share