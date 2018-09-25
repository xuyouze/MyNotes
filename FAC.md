# Paper Name:
  **_Multi-task learning of cascaded CNN for facial attribute classification PR_**

# 1.以前方法的缺陷:
* are based on the fixed loss weight without considering the differences of facial attributes
* usually pre-process the input-image(face detection and alignment)
* ignore the inherent dependency of face detection、 facial landmark localization and face attribute classification

# 2.遇到了什么问题:
~~~
viewpoint、illumination、expression
~~~
# 3.提出了什么:
**_MFCA_** propose a novel multi-task learning of cascaded cnn method ,
use three cascaded sub-networks (i.e., S_Net, M_Net and L_Net corresponding to the neural networks under different scales) to jointly train multiple tasks in a coarse-to-fine manner the proposed method automatically assigns the loss weight to each facial attribute based on a novel dynamic weighting scheme
# 4.方法的细节:
  网络结构是:
  ~~~
    S-Net
      input: 56*56*3
      architecture: VGG16
      output: 1*256*1*1

    M-Net
      input: 112*112*3
      architecture: VGG16
      output: 1*1280*1*1 ( 1024 + S-Net 256)

    L-Net
      input: 224*224*3
      architecture: VGG-16
      output:  1024 + M-Net 1280)
  ~~~


# 5.有什么优势:
* end to end optimization<br/>
* it is the first work to perform multi-task learning in a unified framework for predicting multiple facial attributes simultaneously
* jointly trains different sub-networks in a cascaded manner
* easily apply the back propagation algorithm to train the whole framework.
