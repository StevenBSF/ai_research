这里的

```
return np.linalg.norm(self.x_train - point, axis=1)
```

就是在一步搞定所有训练样本到给定样本 point 的欧氏距离（L2 距离）计算。具体分解一下：

1. **self.x_train 的形状**

   - 假设你有 N 个训练样本，每个样本是 D 维特征向量，那么

     $\texttt{self.x\_train}.shape = (N, D)$

2. **point 的形状**

   - 这是你要预测的单个样本，也是一个 D 维向量：

     $\texttt{point}.shape = (D,)$

3. **向量相减与广播（broadcasting）**

   - self.x_train - point 会把 point 自动“复制”成一个跟 self.x_train 同样行数的数组，然后逐行相减，结果形状仍然是 (N, D)。

   - 数学上就是对每个训练样本 x_i\in\mathbb R^D 做操作

     $x_i - \text{point},\quad i=1,\dots,N.$

4. **np.linalg.norm(..., axis=1)**

   - np.linalg.norm(A, axis=1) 的含义是：对矩阵 A 的每一行（axis=1）计算它的向量范数（默认是二范数，也就是欧氏范数）。

   - 也就是对每行 $\Delta_i = x_i - \text{point}$ 计算

     $\|\Delta_i\|_2 = \sqrt{\sum_{j=1}^D (\Delta_{i,j})^2}.$

   - 最终返回一个长度为 N 的一维数组，每个元素就是对应训练样本与 point 的欧氏距离。

**总结**：这一行代码等价于下面的“显式”写法：

```
dists = []
for x_i in self.x_train:      # x_i 是 (D,) 向量
    delta = x_i - point       # 逐元素相减，(D,)
    dist = np.sqrt((delta**2).sum())
    dists.append(dist)
return np.array(dists)       # shape (N,)
```

只是用 NumPy 的广播和批量运算把这个循环合并到了一行里，效率更高。

下面给出一个最简单的例子，演示如何用 np.linalg.norm(A, axis=1) 计算每一行的欧氏范数。

```
import numpy as np

# 构造一个 3×2 的矩阵 A
A = np.array([
    [3,  4],   # 第 1 行
    [5, 12],   # 第 2 行
    [8, 15]    # 第 3 行
])

# 计算每一行的二范数（Euclidean norm）
dists = np.linalg.norm(A, axis=1)
print(dists)
```

运行结果：

```
[ 5. 13. 17.]
```

**手算验证**

- 第 1 行 [3, 4]：\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5
- 第 2 行 [5, 12]：\sqrt{5^2 + 12^2} = \sqrt{25 + 144} = 13
- 第 3 行 [8, 15]：\sqrt{8^2 + 15^2} = \sqrt{64 + 225} = 17

这样就清楚地展示了 np.linalg.norm(A, axis=1) 是如何依次对矩阵 A 的每一行计算欧氏距离／范数的。

在 NumPy 里，每个维度（dimension）都有一个编号，称作“轴”（axis）：

- 对于二维数组（矩阵）A，
  - 轴 0（axis=0）对应“行”的方向，也就是说沿着 **竖直** 方向把元素聚合；
  - 轴 1（axis=1）对应“列”的方向，也就是说沿着 **水平方向** 把元素聚合。

具体来说，假设有这么一个 2×3 的矩阵：

```
A = np.array([
    [a, b, c],
    [d, e, f]
])
# A.shape == (2, 3)
```

- 如果你对 axis=0 求和 A.sum(axis=0)，

  NumPy 会把同一 **列** 的元素加在一起，结果是长度为 3 的向量：

  [\,a+d,\; b+e,\; c+f\,]

- 如果对 axis=1 求和 A.sum(axis=1)，

  NumPy 会把同一 **行** 的元素加在一起，结果是长度为 2 的向量：

  [\,a+b+c,\; d+e+f\,]

把「沿行（row-wise）」和「沿列（column-wise）」记住就好：

- axis=0 → **纵向**聚合 → 每一列算一次
- axis=1 → **横向**聚合 → 每一行算一次

所以在 np.linalg.norm(A, axis=1) 里，NumPy 就是对 A 的每一 **行** 计算范数，返回一个每行一个结果的数组。

在 NumPy 里，维度（轴）编号是从 0 开始的。对于一个三维数组，比如

```python
import numpy as np

# 构造一个形状为 (2, 3, 4) 的三维数组 A
A = np.arange(24).reshape(2, 3, 4)
# A.shape == (2, 3, 4)
```

- **axis=0**：沿着“第一维”聚合（也就是把索引为 0 的维度压掉），结果形状会是 (3, 4)。

```python
B0 = A.sum(axis=0)
print(B0.shape)  # (3, 4)
```

- **axis=1**：沿着“第二维”聚合（把索引为 1 的维度压掉），结果形状会是 (2, 4)。

```python
B1 = A.sum(axis=1)
print(B1.shape)  # (2, 4)
```

- **axis=2**：沿着“第三维”聚合（把索引为 2 的维度压掉），结果形状会是 (2, 3)。

```python
B2 = A.sum(axis=2)
print(B2.shape)  # (2, 3)
```

### **用**

### **np.linalg.norm**

###  **举例**

- **计算每个 “2×4” 矩阵（第二维和第三维一起） 的范数**

```python
# axis=(1,2) 表示先在第二维和第三维上计算范数，得到一个长度为 2 的向量
norms_per_block = np.linalg.norm(A, axis=(1, 2))
print(norms_per_block.shape)  # (2,)
```

- 这会把每个 A[i, :, :] 看成一个扁平的向量，算出两次范数。

- **计算每个 “3 维向量” 的范数**

  若只指定 axis=1，则对形状 (2,3,4) 的数组，就会把第二维（长度为 3）当作向量分量来计算，针对每个 (i, k)，算出

  \sqrt{\sum_{j=0}^{2} A[i,j,k]^2}

  因此结果形状是 (2, 4)：

```python
norms_axis1 = np.linalg.norm(A, axis=1)
print(norms_axis1.shape)  # (2, 4)
```

同理，如果你用 axis=2，就会把第三维（长度 4）当作向量分量来计算，每个 (i, j) 对应一个范数，结果是 (2, 3)。这样，你可以非常灵活地指定在哪个维度（或哪几维）上做范数/求和等操作。

