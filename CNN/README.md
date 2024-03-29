# Network

显然，这里面的代码git下来是不能直接跑的，但是离跑通距离很小。



- AlexNet：2010年提出的模型，在AlexNet.py文件中以层的结构写出，虽然可以直接调用torch的模型的，但是由于分类类别数目不同，所以搞成了这样。当然代码也是copy来的。

- SPP-Net：SPP网络是为了解决图片尺寸不相同时的问题，虽然传统的处理方法是进行resize等操作，但是可能会导致图形形状的变化，这在形状检测中是一个很大的问题，因而使用SPP网络。虽然作者的论文中对SPP层的原理进行了说明，但是算法细节并不多，因而pad操作并不具有较高的鲁棒性，当bin_num较高时代码会报错：pad应该小于kernel。为了使网络能够使用更高的bin_num，解决办法是先进行pad在进行池化，在SPP-Net.py中的两个SPPLayer类可以看到区别。

  另一个问题是受限于torch的dataloader功能，对于一批次的图片，dataloader会将其拼成一个四维张量，如果图片尺寸不同，显然是不能合成一个张量的，因而batch_size会被限制为1，同时需要自己修改dataloader中的数据调出函数。

  当然，最后效果十分喜人。

  - 由于resize会导致图像的形状发生扭曲从而影响模型的训练，但是解决方法除了SPP-Net，还可以将原图片的特征（宽高比）作为神经元放入全连接层。事实证明，这是有效的，在模型严重缩减的情况下，其收敛速度和损失低于是删减后的SPP-Net的一半。

- VGG16:VGG16相较于AlexNet，虽然深度多了很多，但是感受野(filter)小了很多(3*3)，因而在参数数量上没有太大的变化。

- NIN:Network in Network网络中的网络，其有两个创新性的思想：使用微型的网络层代替卷积层，使用GAP(全局平均池化层)来代替全连接层。

  对于每个mlp块，其主要包含了一层卷积层和两层全连接层（1*1卷积），使得其能提取出更为抽象的特征。

  GAP则取代了全连接层，一是防止全连接层较为复杂导致过拟合，再者减少了模型复杂度，同时令特征直接与类别映射。
  
- ResNet:这东西是一把利器。虽然深度相对于广度对准确度有着更加的影响，但是当神经网络的层数较多时，会出现梯度消失的情况，从而导致分类效果的下降。如果我们还想提高网络的深度，我们不妨在网络层中加入一些shortcut，从而实现很好的效果。

- GoogLeNet:够深，但是参数量却小于AlexNet和VGG16。GoogLeNet的核心思想有以下几个，通过Inception块来进行不同尺度特征的感知并融合。通过1*1卷积来实现特征的提取和参数减少。

  - DenseNet:DenseNet有是一个革新，如果说ResNet是开了捷径，那么DenseNet可以说是每个层都有通往其他层的捷径（实际上是将每一层的输出都作为参数传入下一层）。其实现的细节主要有以下几块：使用BN-ReLu-Conv代替传统的卷积层，Transition layers调整尺寸的输出（通过1\*1卷积实现特征的压缩，2\*2均值池化实现特征的降维），Bottleneck 结构（使用3\*3\*256不如使用1\*1\*64+3\*3\*64+1\*1\*256，虽然输出尺寸相同，但是在此过程中需要的参数量也少了很多）。
