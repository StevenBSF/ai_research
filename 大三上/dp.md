## 矩阵连乘

### 最优子结构证明(5)，状态转移方程(5)

**最优子结构证明**

设对于一个矩阵序列 $A_1, A_2, \dots, A_n$，存在划分 $(A_1, \dots, A_k)(A_{k + 1}, \dots, A_n)$，使得 总计算数 a 最小。$(A_1, \dots, A_k)$ 的乘法次数为 b，$(A_{k +1}, \dots, A_n)$ 的乘法次数为 c，两部分相乘的乘法次数为 d，有 a = b + c + d。
假设 b，c 不是对应子问题的最优解，由于 d 只和 划分点 k 有关，因此不变，那么一定各存在一个子划分，使 b' < b，c' < c，使 a = b + c + d > b' + c' + d，即 a 不是最优解，故 b，c 一定是子问题的最优解，证毕。

**状态转移方程**

设 $f(i, j)$ 表示矩阵 $A_i ... A_j$ 相乘的最小乘法次数，有状态转移方程如下：
$$f(i, j) = \begin{cases}
    \min_{i \leq k < j}(f(i, k) + f(k+1, j) + p[i] * p[k + 1] * p[j + 1]), \quad i < j\\
    0 \quad \text{i = j}
\end{cases}$$

### 代码

```cpp

void dp(int* p, int n, int** f, int** r) 
{
    for (int i = 1; i <= n; i ++ )
        f[i][i] = 0; 

    // 可以从步长的角度理解，必须先求所有两两相乘的最优值才能计算三三相乘的最优值
    for (int r = 2; r <= n; r ++ ) // 步长为 1 为初始化，故从 2 开始
        for (int i = 1; i <= n - r + 1; i ++ ) {
            // 枚举起点，必须保证终点合法，即 i + r - 1 <= n
            int j = i + r - 1;
            f[i][j] = f[i][i] + f[i + 1][j] + p[i - 1] * p[i] * p[j];
            for (int k = i + 1; k < j; k ++ ) {
                // 枚举划分点
                int tmp = f[i][k] + f[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (tmp < f[i][j]) {
                    f[i][j] = tmp;
                    r[i][j] = k;
                }
            }
        }
}
```
### 时间复杂度

$$\sum_{r=2}^n\sum_{i=1}^{n-r+1}(1 + j - i) = \sum_{r=2}^n\sum_{i=1}^{n-r+1}r \\ = \sum_{r=2}^n((n+1)r - r^2) = \frac{n^3 + 3n^2 - 4n}{6} = O(n^3)$$

## 最长公共子序列（找所有）

### 最优子结构证明，状态转移方程

**最优子结构证明**

设序列 $X = \{x_1, x_2, \dots, x_m\}$ 和 $Y = \{y_1, y_2, \dots, y_n\}$ 的最长公共子序列为 $Z = \{z_1, z_2, \dots, z_k\}$ ，则

1. 若 $x_m = y_n$，则 $z_k = x_m = y_n$，则 $Z_{k-1}$ 是 $x_{m-1}$ 和 $y_{n-1}$ 的最长公共子序列
2. 若 $x_m \neq y_n$， 且 $z_k \neq x_m$，则 $Z$ 是 $X_{m-1}$ 和 $Y$ 的最长公共子序列
3. 若 $x_m \neq y_n$，且 $z_k \neq y_n$，则 $Z$ 是 $X$ 和 $Y_{n-1}$ 的最长公共子序列

由此可见，2个序列的最长公共子序列包含了这2个序列的前缀的最长公共子序列。因此，最长公共子序列问题具有最优子结构性质。 

**状态转移方程**

$$f(i, j) = \begin{cases}
    0 \quad &i = 0 \ \text{or} \ j = 0 \\
    f(i - 1, j - 1) + 1 \quad &i, j > 0; x_i = y_j \\
    max(f(i - 1, j), f(i, j - 1)) \quad &i, j > 0; x_i \neq y_j 
\end{cases}$$

### 代码

```cpp
void LCS(int m, int n, char* x, char* y, int** f, int** r)
{
    for (int i = 0; i <= m; i ++ ) f[i] = 0;
    for (int i = 0; i <= n; i ++ ) f[i] = 0;

    for (int i = 1; i <= m; i ++ ) 
        for (int j = 1; j <= n; j ++ ) {
            if (x[i] == y[j]) {
                f[i][j] = f[i - 1][j - 1] + 1;
                r[i][j] = 0;
            } else {
                if (f[i - 1][j] >= f[i][j - 1]) {
                    f[i][j] = f[i - 1][j];
                    r[i][j] = 1;
                } else {
                    f[i][j] = f[i][j - 1];
                    r[i][j] = 2;
                }
            }
        }
}
```

求解所有可能解需要对原记录做些修改，即单独记录f相等的状态
```cpp
void LCS(int m, int n, char* x, char* y, int** f, int** r)
{
    for (int i = 0; i <= m; i ++ ) f[i] = 0;
    for (int i = 0; i <= n; i ++ ) f[i] = 0;

    for (int i = 1; i <= m; i ++ ) 
        for (int j = 1; j <= n; j ++ ) {
            if (x[i] == y[j]) {
                f[i][j] = f[i - 1][j - 1] + 1;
                r[i][j] = 0;
            } else {
                if (f[i - 1][j] >= f[i][j - 1]) {
                    f[i][j] = f[i - 1][j];
                    r[i][j] = 1;
                    if (f[i - 1][j] == f[i][j - 1]) r[i][j] = 3;
                } else {
                    f[i][j] = f[i][j - 1];
                    r[i][j] = 2;
                }
            }
        }
}

void trace(int m, int n, char* x, char* y, int** r, std::string cur, std::vector<std::string>& res)
{
    if (m == 0 || n == 0) {
        res.push_back(cur);
        return;
    }

    if (r[m][n] == 0) trace(m - 1, n - 1, x, y, r, x[m] + cur, res);
    else { // 这里简化了一下，因为 3 需要遍历 1 2 两种状态，即 不是 1 必定要 trace(m - 1, n)， 不是 2 必定要 trace(m, n - 1)
        if (r[m][n] != 1) trace(m, n - 1, x, y, r, cur, res);
        if (r[m][n] != 2) trace(m - 1, n, x, y, r, cur, res);
    }
}
```

### 时间复杂度

$$\sum_{i=1}^m\sum_{j=1}^n\frac{3 + 4 + 4}{3} = \frac{11}{3}mn = O(mn)$$

## 最大子段和（倒着找）

### 最优子结构证明，状态转移方程

**最优子结构证明**

设 $f(n)$ 是以下标为 $n$ 的元素为结尾时的最大子段和，那么该问题的最优解为 

$$\max_i f(i) \quad 1 \leq i \leq n \\ f(i) = \max_j\sum_{k=j}^ia_k \quad 1 \leq j \leq i$$

因此只需要证明问题：以下标为 $n$ 的元素为结尾时的最大子段和满足最优子结构即可证明原问题有最优子结构

1. 最优解包含 $a_{n-1}$
设 $f(n)$ 的最优解为 $i$，那么 $i$ 也是子问题 $f(n-1)$ 的最优解，即

$$f(n) = \sum_{k=i}^{n}a_k \\ f(n-1) = \sum_{k=i}^{n-1}a_k$$

否则，设 $j$ 是子问题的最优解, 而 $i$ 不是其最优解，有

$$\sum_{k=j}^{n-1}a_k > \sum_{k=i}^{n-1}a_k$$

故

$$\sum_{k=j}^{n-1}a_k + a_n > \sum_{k=i}^na_k$$

说明存在更优解 $j$，矛盾

2. 最优解不包含 $a_{n-1}$
此时 $f(n) = a_n$，和子问题无关

综上满足最优子结构

**状态转移方程**

如果最优解包含 $a_{n-1}$，那一定有 $f_{n-1} + a_n > a_n$，即 $f_{n-1} > 0$

$$f(i) = \begin{cases}
    a_i \quad &f_{i-1} \leq 0 \\
    f_{i-1} + a_{i} \quad &f_{i-1} > 0
\end{cases}$$

### 代码

```cpp
void MaxSubsequence(int n, int* a, int* f, int* r)
{
    f[0] = 0;
    int maxx = 0;
    for (int i = 1; i <= n; i ++ ) {
        if (f[i - 1] > 0) f[i] = f[i - 1] + a[i];
        else f[i] = a[i];
        maxx = std::max(maxx, f[i]);
    }
}
```

f(i) 只依赖于 f(i-1) 可以做如下更改：
```cpp
void MaxSubsequence(int n, int* a)
{
    int f = a[1];
    int maxx = f;
    for (int i = 2; i <= n; i ++ ) {
        if (f > 0) f = f + a[i];
        else f = a[i];
        maxx = std::max(maxx, f);
    }
}
```

记录最大子段的位置，如果题目要求起始点和结束点都靠前可以倒序递推

```cpp
void MaxSubsequence(int n, int* a)
{
    int f = a[n-1];
    int maxx = f;
    int tmp, B, E;
    tmp = B = E = n - 1;

    for (int i = n - 2; i >= 1; i -- ) {
        if (f > 0) f = f + a[i];
        else {
            f = a[i];
            tmp = i;  // 记录新的结束点
        }
        if (maxx <= f) {
            B = i;   // f(i) 现在表示以 i 为开头的最大子段和
            E = tmp;
            maxx = f;
        }
    }
}
```

### 时间复杂度 

$$\sum_{i=1}^n\frac{3 + 3}{2} = 3n = O(n)$$

## 01背包（优化不考）

### 最优子结构证明，状态转移方程

**最优子结构证明**

设 $(y_1, y_2, \dots, y_n)$ 是所给 0-1 背包的一个最优解，则 $(y_2, y_3, \dots, y_n)$ 是下面对应子问题的一个最优解

$$max\sum_{i=2}^nv_ix_i = \begin{cases}
    \sum_{i=2}^nw_ix_i \leq c - w_1y_1 \\
    x_i \in \{0, 1\}
\end{cases}$$

否则，设 $(z_2, z_3, \dots, z_n)$ 是上述问题的一个最优解，而 $(y_2, y_3, \dots, y_n)$ 不是它的最优解，有

$$\sum_{i=2}^nv_iz_i > \sum_{i=2}^nv_iy_i \\ w_1y_1 + \sum_{i=2}^nw_iz_i \leq c$$

故 

$$v_1y_1 + \sum_{i=2}^{n}v_iz_i > \sum_{i=1}^nv_iy_i \\ w_1y_1 + \sum_{i=2}^nw_iz_i \leq c$$

说明存在更优解 $(y_1, z_2, \dots, z_n)$ 使得 $(y_1, y_2, \dots, y_n)$ 不是最优解，矛盾。

**状态转移方程**

这里从后往前是因为证最优子结构时子问题是不考虑第一个物品时的问题

$$f(i, j) = \begin{cases}
    max(f(i + 1, j - w_i) + v_i, f(i + 1, j)) \quad &w_i \geq j \\ f(i + 1, j) \quad &w_i < j
\end{cases}$$

$$f(n, j) = \begin{cases}
    v_n \quad &j \geq w_n \\ 0 \quad &j < w_n
\end{cases}$$

### 代码

```cpp
void Knapsack(int n, int c, int* v, int* w, int** f) {
    for (int i = 0; i <= c; i ++ ) f[n+1][i] = 0;
    for (int i = n; i >= 1; i -- ) 
        for (int j = 0; j <= c; j ++ ) {
            f[i][j] = f[i+1][j];
            if (j >= w[i]) f[i][j] = std::max(f[i][j], f[i+1][j-w[i]] + v[i]);
        }
}
```

### 时间复杂度

$$\sum_{i=1}^n\sum_{j=0}^{c}2 = 2n(c+1) = O(nc)$$