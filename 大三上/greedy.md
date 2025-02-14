## 活动安排

**贪心策略**

将活动按照结束时间最早到最晚排列，第一次选择结束时间最早的活动，之后每次选择结束最早且与之前活动相容的活动，即 $s_i > f_j$
**贪心选择证明：**

设活动序列为 $E = (1, 2, 3, \dots, n)$
1. 设有某最优安排 A = $(k, x_2, x_3, ) \in E$，A中的第一个活动为k
   1. 如果 k = 1
   则 A 是一个以贪心选择开始的最优解
   2. 如果 k > 1
   设序列为 $B = (k, x_2, _3, \dots, x_n)$，由相容性，一定有 $s(x_2) > f(k)$，那么此时如果将活动替换成 活动1，有 $s(x_2) > f(1)$，故活动1 也和 $(x_2, x_3, \dots, x_n)$ 相容，故如果 $k \neq 1$，可以替换成以活动 1 开头的序列，二者活动总数相同，则 A 仍是最优的，故总是存在以贪心选择开始的最优方案
2. 对问题 $E = (1, 2, 3, \dots, n)$，按照贪心策略，选择结束时间最早的活动 1，子问题化简为 
    $$E' = \{i \in E | s_i > f_1\}$$
    设存在 E' 的另一个最优解 B'，有 $\sum_{k \leq }$，那么由于 $\forall i \in E'，s_i > f_1$，可以将 活动1 加入 B'，即，对于问题 E，存在更优解 $|B| = |B'| + 1 > |A'| + 1 = |A|$，与 A 是 E 的最优解矛盾


```cpp
void Sort(int* arr, int* index, int n)
{
    for (int i = 1; i <= n; i ++ ) index[i] = i;
    std::sort(index + 1, index + n + 1, [&](const auto& a, const auto& b) {
        return arr[a] < arr[b];
    });
}

void Arrange(int n, int* s, int* f, bool* x)
{
    int* t = new int[n + 1];
    Sort(f, t, n);

    x[t[1]] = true;
    int j = t[1];
    
    for (int i = 2; i <= n; i ++ ) {
        if (s[t[i]] >= f[j]) {
            x[t[i]] = true;
            j = t[i];
        } else x[t[i]] = false;
    }
}
```

## 最优装载问题

**贪心策略**
每次迭代，从剩余集装箱中选择最轻的货物

**贪心选择证明：**
设集装箱已按重量从小到大排序，$X = (x_1, x_2, \dots, x_n)$ 是最优装载问题的最优解
设最优解中，k 为最先装入箱中的最轻货物，即 $$k = min_{1\leq i \leq n}(i | x_i = 1)$$

1. 以贪心选择开始的最优解
   1. k = 1 时
    第一个最轻货物被装入，货物放入顺序满足贪心选择策略，X 是一个以贪心选择开始的最优解
   2. k > 1 时
    此时的解向量满足约束 $\sum_{1\leq i \leq n}x_iw_i \leq c$，又因为第一个货物的重量小于等于 k，$w_1 \leq w_k$，那么，将 k 替换为 1 得到的解向量 $Y = (y_1, y_2, \dots, y_n)$，$\sum_{1 \leq i \leq n}y_iw_i \leq \sum_{1\leq i \leq n}x_iw_i \leq c$，约束仍然满足，且有两个解的货物数量一致，即 $\sum_{1 \leq i \leq n}y_i = \sum_{1 \leq i \leq n}x_i$，故总是存在以贪心选择开始的最优方案
2. 按照贪心选择策略，子问题的解向量化简为 $X' = (x_2, x_3, \dots, x_n)$，$c' = c - w_1$
    设存在 更优解 $Y' = (y_2, y_3, \dots, y_n)$，那么其满足约束 $$\sum_{2 \leq i \leq n}y_iw_i \leq c - w_1$$ 此时，将货物 1 加入 Y'，可知 $w_1 + \sum_{2\leq i \leq n}\leq c - w_1 + w_1 = c$，满足原问题约束，故存在更优解 $X' = (1, y_2, y_3, \dots, y_n)$，$1 + sum_{2 \leq i \leq n}y_i \geq 1 + \sum_{2 \leq i \leq n}x_i$，与 X 是最优解相矛盾

```cpp
void Sort(int* arr, int* index, int n)
{
    for (int i = 1; i <= n; i ++ ) index[i] = i;
    std::sort(index + 1, index + n + 1, [&](const auto& a, const auto& b) {
        return arr[a] < arr[b];
    });
}

void Loading(int n, int* w, int c, bool* x)
{
    int* t = new int[n + 1];
    Sort(w, t, n);

    for (int i = 1; i <= n; i ++ ) x[i] = false;

    for (int i = 1; i <= n; i ++ ) {
        if (c >= w[t[i]]) {
            x[t[i]] = true;
            c -= w[t[i]];
        } else break;
    }
}
```

## 单源最短路径

**贪心策略**
每次迭代，从 $V-S_i$ 中选择具有最短特殊路径 dist[u] 的顶点 u，加入 $S_i$ 得到 $S_{i+1}$

**贪心选择证明：**
假设顶点 u 是集合 S 中的第一个满足 $d(v, u) < dist[u]$ 的顶点，即 $d(v, u) \neq dist[u]$，且全局最优路径经过S之外的顶点
从 v 到 u 的全局最短路径上，经过的第 1 个属于 $V-S_i$ 的顶点为 x
对 v 到 u 的全局最短路径，有 d(v, x) + distance(x, u) = d(v, u)，由于 distance(x, u) > 0 有 d(v, x) < d(v, u)
由于 d(v, u) < dist[u]，有 d(v, x) + distance(x, u) < d(v, u) < dist[u]
那么有 $dist[x] = d(v, x) \leq d(v, x) + distance(x, u) = d(v, u) < dist[u]$，即 dist[x] < dist[u]
但根据贪心策略，每次都会选择 dist 最小的顶点加入 $S_i$ 中，即 $dist[u] \leq dist[x]$，相矛盾

```cpp
void Dijkstra(int* dist, bool* visit, int* cost, int s, int n)
{
    for (int i = 1; i <= n; i ++ ) {
        visit[i] = false;
        dist[i] = INF;
    }

    dist[s] = 0;
    visit[s] = true;

    for (int i = 1; i < n; i ++ ) {
        int temp = INF;
        int u = s;
        for (int j = 1; j <= n; j ++ ) 
            if (!visit[j] && temp > dist[j]) {
                temp = dist[j];
                u = j;
            }
        visit[u] = true;

        for (int j = 1; j <= n; j ++ ) 
            if (!visit[j] && cost[u][j] < INF) {
                dist[j] = std::min(dist[u] + cost[u][j], dist[j]);
            }
    }
}
```

## 背包问题

**贪心策略**
每次迭代选择价值 v / w 最高的物品放入背包，如果不能完全放入，尽可能放入背包

**贪心选择证明：**

1. 设数组 v, w, 物品编号已经根据价值从高到低重排列，一个最优解的解向量为 $X = (x_1, x_2, \dots, x_n)$
    设第一个放入的物品编号为 $k = min_{1 \leq i \leq n}(i | x_i = 1)$
    1. k = 1
        此时选择的是价值最高的物品，是以贪心选择为开始的最优解
    2. k > 1
        此时解向量满足约束 $\sum_{1 \leq i \leq n}{w_ix_i} \leq c$，由题可知，$v_1/w_1 > v_k/w_k$，此时，有
        1. 如果 $w_1 \geq w_k$
            将 k 全部替换为 1，有 $\frac{w_k}{w_1}v_1 + \sum_{k < i \leq n}v_ix_i \geq \sum_{1 \leq i \leq n}v_ix_i$，总重量不变，仍满足约束，即存在更优解如上所述大于当前最优解，相矛盾 
        2. $w_1 < w_k$
            将 k 中 $w_1$ 的物品换为 1 中物品，有 $\frac{w_1}{w_k}v_1 + \frac{w_k - w_1}{w_k}v_k + \sum_{k < i \leq n}v_ix_i \geq \sum_{1 \leq i \leq n}v_ix_i$，总重量不变，仍满足约束，即存在更优解如上所述大于当前最优解，相矛盾
            综上，不存在以 k > 1 作为第一个物品的最优解，故一定是以贪心选择开始
2. 设当前问题的解向量为 $X = (x_1, x_2, \dots, x_n)$，其子问题的解向量为 $X' = (x_2, x_3, \dots, x_n)$，且满足 $\sum_{2 \leq i \leq n}x_iw_i \leq c - w_1$，由于当 c < w_1 时该问题没有子问题，此处不讨论
    假设子问题存在一个更优解 $Y' = (y_2, y_3, \dots, y_n)$，使得 $\sum_{2 \leq i \leq n}w_iy_i \leq c - w_1$, $\sum_{2 \leq i \leq n}v_iy_i > \sum_{2 \leq i \leq n}v_ix_i$，此时把 1 放入背包，此时原问题仍满足约束 $w_1 + \sum_{2 \leq i \leq n}w_iy_i \leq c - w_1 + w_1 = c$，且 $w_1v_1 + \sum_{2 \leq i \leq n}v_iy_i > \sum_{1 \leq i \leq n}v_ix_i$，即存在更优解 $Y = (1, y_2, y_3, \dots, y_n)$ 优于最优解 X，相矛盾

```cpp
void Sort(int* arr, int* brr, int* index, int n)
{
    for (int i = 1; i <= index; i ++ ) index[i] = i;
    std::sort(index + 1, index + n + 1, [&](const auto& a, const auto& b) {
        return arr[a] / brr[a] > arr[b] / brr[b];
    })
}

void Knapsack(int n, float c, float* v, float* w, bool* x)
{
    int* t = new int[n + 1];
    Sort(v, w, t, n); 
    for (int i = 1; i <= n; i ++ ) x[i] = false;

    for (int i = 1; i <= n; i ++ ) {
        if (w[t[i]] <= c) {
            c -= w[t[i]];
            x[t[i]] = true;
        } else if (c > 0) {
            x[t[i]] = c / w[t[i]];
            break;
        }
    }
}
```