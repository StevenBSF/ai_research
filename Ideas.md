- 我能不能尝试构建causal graph监督mvc？





## 用来对比的论文框架图

- Multi-VAE（ICCV 2021）
  - ![image-20240830150031291](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830150031291.png)

- GCFAgg(CVPR 2023)
  - ![image-20240830143059272](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830143059272.png)

- ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK(ICLR 2021)
  - ![image-20240830143159431](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830143159431.png)
- (CTCC)Cross-view Topology Based Consistent and Complementary Information for Deep Multi-view Clustering(ICCV 2023)
  - ![image-20240830143513186](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830143513186.png)
- (DFP-GNN)( IEEE TRANSACTIONS ON MULTIMEDIA 2023)
  - ![image-20240830143747634](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830143747634.png)
- MFLVC(CVPR 2022)
  - ![image-20240830143927798](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830143927798.png)
- Robust Multi-View Clustering With Incomplete Information（TPAMI 2023）
  - ![image-20240830144840287](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830144840287.png)
- SDMVC（IEEE Transactions on Knowledge and Data Engineering 2023）
  - ![image-20240830145812178](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830145812178.png)
- MIMC(CIKM,ccf b类)
  - ![image-20240830144450580](C:\Users\12895\AppData\Roaming\Typora\typora-user-images\image-20240830144450580.png)

- 心得
  - MFLVC和SDMVC都是从挖掘语义信息作聚类和作伪标签图的两个角度监督聚类效果。
  - 对于语义信息的挖掘，GCFAgg仿照transformer进行特征挖掘
  - DealMVC的融合我认为参考了ADAPTIVE UNIVERSAL GENERALIZED PAGERANK GRAPH NEURAL NETWORK这篇文章？并且还有可以修改的地方？
  - 互信息是一个可以找idea的点
  - 在特征层面作扰动
  - 对于不同视角下的特征，各种拼接方法很有意思，可以作为切入点