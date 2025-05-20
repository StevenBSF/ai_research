## DeCLIP:supervision exists everywhere:a data efficient contrastive language-image pre-training paradigm

个人感觉这篇论文就是堆积已有的自监督方法，一般来说约束越多效果就越好。卡太少的话，不建议，跑不动呀

- motivation：

1. 论文是为了充分利用单模态和多模态，充分利用单模态特征用自监督进行实验，多模态用图像文本对实现；
2. 一个图片的文本描述大部分都是集中在局部区域，作者使用RRC得到一个图像的局部区域进行实现；
3. 一个图片有多种描述方式，提出用最近邻文本检索得到更多文本监督。（i.e.,对图像的文本描述1的特征向量在队列库中求余弦相似性得到最相似的描述2）

在SLIP基础上新增一个文本域的自监督，即该论文使用图片自监督+文本自监督+2*2的图像-文本对的对比监督。

- DeCLIP是SLIP的进化版：

1. 图像自监督：使用[SimSiam](https://zhida.zhihu.com/search?content_id=218307524&content_type=Article&match_order=1&q=SimSiam&zhida_source=entity)，最大化一个图像的两种增强后所得的特征；
2. 文本自监督：采用[masked 语言模型](https://zhida.zhihu.com/search?content_id=218307524&content_type=Article&match_order=1&q=masked+语言模型&zhida_source=entity)；
3. 图像-文本模态：原始的 CLIP 不使用文本增强，仅使用随机方形裁剪图像增强，因此需要大量数据。deCLIP使用随机数据增强，相比于原始CLIP，该论文多了3倍的监督信息。
4. 在嵌入空间中使用了最近邻监督更好利用相似性的文本信息。论文维护一共队列空间，在嵌入空间中使用最近邻检索得到最相似的文本描述，然后使用文本监督loss得到额外的监督。



![img](https://pic4.zhimg.com/v2-6e71e6507f3ac239466a8c9c071c7821_1440w.jpg)

CLIP与DeCLIP逻辑对比图

- 图像自监督框架：



![img](https://pic1.zhimg.com/v2-1c48c76afddb8d702205a0adce5961a4_1440w.jpg)

图片对比自监督



- 文本自监督框架：每个句子中随机选择15%的单词，然后，80%的时间用【mask】替换单词，用10%的时间用随机token替换单词，用10%的时间不改变单词。最后得到语言模型对应的token域原始token进行交叉熵loss。

![img](https://pic2.zhimg.com/v2-b7864c93b327bf4b6ffce441be6652b7_1440w.jpg)