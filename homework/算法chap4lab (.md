![image-20241108133233157](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108133233157.png)

## ***计算机学院（国家示范性软件学院）***

# 算法设计与分析实验报告

## 贪心算法

<center>姓名：包诗峰</center>
<center>学号：2022211656</center>
<center>班级：2022211301</center>

<div STYLE="page-break-after: always;"></div> 







# Huffman编码



**1. 实验内容**

**1.1 题目描述**

给定 N个字符及其出现的概率 ，通过 Huffman 编码生成一棵二叉树，计算该树的加权路径长度（期望编码长度）。使用优先队列构建 Huffman 树，并计算每个字符的编码期望值。

**1.2 输入格式**

输入文件名为 huffman.in，输入共两行。

​	•	第一行包含一个整数 ，表示字符的个数。

​	•	第二行包含 N个小数 ，分别表示每个字符的出现概率。



**1.3 输出格式**

输出文件名为 huffman.out，输出共一行。

​	•	第一行包含一个小数，为 Huffman 编码的期望值，保留三位小数。



**1.4 输入输出样例**

输入：

```C++
4	1.600
0.1 0.1 0.2 0.6	
```

输出：

```c++
1.600
```

**1.5 数据范围**

​	•	 $1 \leq n \leq 10^6$ 

​	•	 $0 < p_i \leq 1$ 

​	•	$\sum_{i=1}^n p_i = 1$

**2. 实验设计思路**

在 Huffman 编码问题中，给定 $n$个字符及其出现概率 ，目标是通过构造 Huffman 树，使得总的加权路径长度（即期望编码长度）最小。

**2.1 状态定义**

​	•	使用一个优先队列（小根堆）存储所有字符的出现概率。

​	•	每次从队列中取出两个最小的概率值合并，计算累加的代价，并将合并后的值重新插入优先队列。

**2.2 计算过程**

​	1.	初始化优先队列，将所有字符的概率值插入。

​	2.	每次从队列中取出两个最小值 $a$和$b$ ，计算其合并代价$a+b$ ，并将合并后的值重新插入队列。

​	3.	累加合并代价，直到队列中仅剩一个值为止。

​	4.	最终累加代价即为 Huffman 编码的期望值。

**2.3 算法复杂度分析**

​	•	**时间复杂度**：优先队列的插入与删除操作的时间复杂度为$O(\log n) $，共进行$n-1$次合并操作，故总时间复杂度为$O(n \log n) $ 。

​	•	**空间复杂度**：使用优先队列存储$n$个概率值，空间复杂度为$O(n)$。



**3.1 代码实现**

```c++
#include <iostream>
#include <queue>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n;
    cin >> n;

    // 小根堆
    priority_queue<double, vector<double>, greater<double>> pq;

    for (int i = 0; i < n; ++i) {
        double p;
        cin >> p;
        pq.push(p);
    }

    // 计算 Huffman 编码期望
    double totalCost = 0.0;
    while (pq.size() > 1) {
        double a = pq.top(); pq.pop();
        double b = pq.top(); pq.pop();
        totalCost += a + b;
        pq.push(a + b);
    }

    // 输出结果，保留三位小数
    cout << fixed << setprecision(3) << totalCost << "\n";
    return 0;
}
```





**4. 实验结果**

**4.1 测试用例 1**

输入：

```c
4
0.1 0.1 0.2 0.6
```

输出：

```c
1.600
```



**4.2 测试用例 2**

输入：

```c
6
0.05 0.1 0.15 0.2 0.25 0.25
```

输出：

```c
2.200
```

**4.3 测试用例 3**

输入：

```c
10
0.05 0.05 0.1 0.1 0.1 0.1 0.15 0.15 0.1 0.1
```

输出：

```c++
3.500
```

**5. 实验总结**

​	1.	**实验结果**：通过 Huffman 编码构建最优二叉树，程序能够在大规模输入（如 ）时高效计算总期望。

​	2.	**改进方向**：

​	•	当前实现使用小根堆，适合大数据量的输入，具有较高的时间和空间效率。

​	•	若输入数据较小，可以进一步优化算法以简化实现。

​	3.	**实践收获**：

​	•	掌握了优先队列的使用。

​	•	理解了 Huffman 编码的核心思想：通过贪心策略最小化加权路径长度。

​	•	验证了算法的时间复杂度和空间复杂度。



# dijkstra最短路



**1. 实验内容**

**1.1 题目描述**

给定一个无向图，图中包含 V 个顶点和 E 条边，求解从顶点 1 到目标顶点 V 的最短路径权值之和。如果顶点 1 和顶点 V 不连通，则输出 -1。

**1.2 输入格式**

输入文件名为 dijkstra.in，输入共 $E+1$ 行。

​	•	第一行包含两个整数 $V$ 和 $E$，分别表示顶点数和边数。

​	•	接下来的 $E$ 行，每行包含三个整数 $u$, $v$, $w$，表示无向边 $u$ 和 $v$ 之间的权值为 $w$。

**1.3 输出格式**

输出文件名为 dijkstra.out，输出共一行。

​	•	第一行包含一个整数，为从顶点 1 到顶点 $V$ 的最短路径权值之和。如果两点不连通，则输出 -1。

**1.4 输入输出样例**

输入：

```c
3 3
1 2 5
2 3 5
3 1 2
```

输出：
```c
2
```

**1.5 数据范围**

​	•	 $0 < V \leq 5000$ 

​	•	 $0 < E \leq 2 \times 10^5$ 

​	•	 $0 < u, v \leq V$ 

​	•	 $0 < w \leq 2 \times 10^5$ 

**2. 实验设计思路**



本实验采用 **Dijkstra 算法** 计算从源点（顶点 1）到目标点（顶点 V）的最短路径。Dijkstra 算法是经典的单源最短路径算法，适用于边权为非负的图。



**2.1 核心思想**



Dijkstra 算法的核心思想是使用 **贪心策略**，每次选择距离源点最近的未处理节点，将其加入最短路径集合，同时更新其邻接节点的最短路径。



**2.2 算法流程**

​	1.	**初始化**：

​	•	使用一个距离数组 dist，初始化所有节点的最短距离为无穷大（INF），源点（顶点 1）的距离为 0。

​	•	使用优先队列（小根堆）维护当前未处理的节点及其到源点的距离。

​	2.	**更新最短路径**：

​	•	每次从堆中取出距离源点最近的节点 u，更新其所有邻接节点 v 的最短距离。如果通过 u 到达 v 的距离更短，则更新 dist[v]。

​	3.	**终止条件**：

​	•	如果目标节点 $V$ 被处理，则直接返回其最短距离。

​	•	如果堆为空且目标节点未被访问，则说明目标节点不可达，输出 -1。



**2.3 算法复杂度**

​	•	**时间复杂度**：$O((V + E) \log V)$，主要由优先队列的操作复杂度决定。

​	•	**空间复杂度**：$O(V + E)$，存储图的邻接表和距离数组。

**3.1 代码实现**

```c++
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

const int INF = numeric_limits<int>::max();

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);


    int V, E;
    cin >> V >> E;

    vector<vector<pair<int, int>>> graph(V + 1); // 邻接表，graph[u] 存储 (v, w)
    for (int i = 0; i < E; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        graph[u].emplace_back(v, w);
        graph[v].emplace_back(u, w); // 无向图
    }

    int target = V;

    // Dijkstra 初始化
    vector<int> dist(V + 1, INF);
    dist[1] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq; // 小根堆 (dist, node)
    pq.emplace(0, 1);

    // Dijkstra 算法
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue; // 如果不是最优距离，跳过

        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.emplace(dist[v], v);
            }
        }
    }

    // 输出结果
    if (dist[target] == INF) {
        cout << -1 << "\n";
    } else {
        cout << dist[target] << "\n";
    }

    return 0;
}
```



**4. 实验结果**

**4.1 测试用例 1**

输入：

```c++
3 3
1 2 5
2 3 5
3 1 2
```

输出：

```c++
2
```

**4.1 测试用例 2**

输入：

```c++
4 4
1 2 3
1 3 2
3 4 4
2 4 6
```

输出：

```c++
6
```

**4.1 测试用例 3**

输入：

```c++
4 2
1 2 5
3 4 7
```

输出：

```c++
-1
```

**5. 实验总结**

​	1.	**实验结果**：Dijkstra 算法能够正确求解从源点到目标点的最短路径，尤其在稠密图和稀疏图中表现良好。

​	2.	**改进方向**：

​	•	当前算法适用于非负边权的图，若需支持负边权图，可考虑使用 Bellman-Ford 算法。

​	•	可以结合 A* 算法提高目标导向的性能。



# 最小生成树 1(prim)

**1. 实验内容**

**1.1 题目描述**

给定一个无向图，其中包含 V 个顶点和 E 条边，边的权重为非负值。通过 **Prim 算法**，求以顶点 1 为根的最小生成树（MST）的总权值。如果图不连通，则输出 -1。

**1.2 输入格式**

输入文件名为 prim.in，输入共 E+1 行。

​	•	第一行包含两个整数 V 和 E，分别表示顶点数和边数。

​	•	接下来的 E 行，每行包含三个整数 u, v, w，表示无向边 u 和 v 的权重为 w。

**1.3 输出格式**

输出文件名为 prim.out，输出共一行。

​	•	第一行包含一个整数，为最小生成树的总权值。如果图不连通，则输出 -1。

**1.4 输入输出样例**

输入：

```c
4 5
1 2 2
1 3 2
1 4 3
2 3 4
3 4 3
```

输出：
```c
7
```

**1.5 数据范围**

​	•	 $0 < V \leq 5000$ 

​	•	 $0 < E \leq 2 \times 10^5$ 

​	•	 $0 < u, v \leq V$ 

​	•	 $0 < w \leq 2 \times 10^5$ 

**2. 实验设计思路**

本实验采用 **Prim 算法** 求解无向图的最小生成树。Prim 算法通过逐步扩展 MST，从初始顶点开始，每次选择权值最小且不形成环的边加入 MST，直至所有顶点都被覆盖。

**2.1 核心思想**

​	1.	使用一个小根堆（优先队列）来维护当前可访问边的权值。

​	2.	每次从优先队列中选取权值最小的边，将其对应的节点加入 MST。

​	3.	更新与新增节点相邻的边，重复上述过程直到所有节点都被访问。

**2.2 算法流程**

​	1.	**初始化**：

​	•	使用一个布尔数组 inMST 标记节点是否已加入 MST。

​	•	使用一个数组 minWeight 记录每个节点到 MST 的最小边权值。

​	•	初始化小根堆 pq，从顶点 1 开始，将其边权值设为 0。

​	2.	**迭代更新**：

​	•	从优先队列中取出权值最小的边，将对应节点加入 MST。

​	•	更新新加入节点的所有邻接节点，更新这些节点到 MST 的最小边权值。

​	3.	**终止条件**：

​	•	当所有节点都被加入 MST 时，算法结束，输出 MST 的总权值。

​	•	如果有节点未被访问，则输出 -1。

**2.3 算法复杂度**

​	•	**时间复杂度**：$O((V + E) \log V)$，主要由优先队列的操作复杂度决定。

​	•	**空间复杂度**：$O(V + E)$，存储图的邻接表和辅助数组。

**3.1 代码实现**

```c++
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

const int INF = numeric_limits<int>::max();

int main() {

    int V, E;
    cin >> V >> E;

    vector<vector<pair<int, int>>> graph(V + 1); // 邻接表
    for (int i = 0; i < E; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        graph[u].emplace_back(v, w);
        graph[v].emplace_back(u, w); // 无向图
    }

    // Prim 初始化
    vector<bool> inMST(V + 1, false); // 记录节点是否已加入 MST
    vector<int> minWeight(V + 1, INF); // 节点到 MST 的最小边权重
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq; // 小根堆 (权重, 节点)

    minWeight[1] = 0;
    pq.emplace(0, 1); // 从节点 1 开始

    int mstCost = 0; // 最小生成树的总权重

    // Prim 算法
    while (!pq.empty()) {
        auto [weight, u] = pq.top();
        pq.pop();

        if (inMST[u]) continue; // 节点已在 MST 中，跳过
        inMST[u] = true;        // 加入 MST
        mstCost += weight;      // 累加权重

        for (auto [v, w] : graph[u]) {
            if (!inMST[v] && w < minWeight[v]) {
                minWeight[v] = w;
                pq.emplace(w, v);
            }
        }
    }

    // 检查是否所有节点都被覆盖
    for (int i = 1; i <= V; ++i) {
        if (!inMST[i]) {
            cout << -1 << "\n"; // 图不连通
            return 0;
        }
    }

    cout << mstCost << "\n";
    return 0;
}
```

**4. 实验结果**

**4.1 测试用例 1**

输入：

```c++
4 5
1 2 2
1 3 2
1 4 3
2 3 4
3 4 3
```

输出：

```c++
7
```

**4.1 测试用例 2**

输入：

```c++
3 2
1 2 5
2 3 7
```

输出：

```c++
12
```

**4.1 测试用例 3**

输入：

```c++
4 2
1 2 5
3 4 7
```

输出：

```c++
-1
```





# 最小生成树 2（kruskal）

**1. 实验内容**

**1.1 题目描述**

使用 **Kruskal 算法** 计算无向图中以顶点 1 为根的最小生成树的边权值和。如果图不连通，则输出 -1。

**1.2 输入格式**

输入文件名为 kruskal.in，输入共 E+1 行。

​	•	第一行包含两个整数 V 和 E，分别表示顶点数和边数。

​	•	接下来的 E 行，每行包含三个整数 u, v, w，表示无向边 u 和 v 的权重为 w。

**1.3 输出格式**

输出文件名为 kruskal.out，输出共一行。

​	•	第一行包含一个整数，为最小生成树的总权值。如果图不连通，则输出 -1。

**1.4 输入输出样例**

输入：

```c
4 5
1 2 2
1 3 2
1 4 3
2 3 4
3 4 3
```

输出：

```c
7
```

**1.5 数据范围**

​	•	 $0 < V \leq 5000$ 

​	•	 $0 < E \leq 2 \times 10^5$ 

​	•	 $0 < u, v \leq V$ 

​	•	 $0 < w \leq 2 \times 10^5$ 

**2. 实验设计思路**

本实验采用 **Kruskal 算法** 求解最小生成树（MST）。Kruskal 算法的核心思想是：

​	1.	按边权值从小到大排序。

​	2.	依次尝试加入边，若加入边不会形成环，则将其纳入 MST。

​	3.	使用 **并查集 (Union-Find)** 数据结构来判断两个顶点是否已经连通。



**2.1 核心思想**

​	1.	**边的排序**：

​	•	按权值升序对所有边进行排序。

​	2.	**判断连通性**：

​	•	使用并查集实现高效连通性判断。若两个顶点的根节点不同，则将边加入 MST。

​	3.	**终止条件**：

​	•	当 MST 中的边数达到 V-1 时，算法结束。

​	•	如果无法覆盖所有顶点，则图不连通，输出 -1。



**2.2 算法流程**

​	1.	**初始化**：

​	•	创建并查集，初始化所有顶点为独立集合。

​	•	按权值升序排序边集。

​	2.	**逐边检查**：

​	•	对于每条边，检查其两个顶点是否在同一集合：

​	•	如果不在同一集合，则将边加入 MST，并合并两个集合。

​	•	如果在同一集合，则跳过该边。

​	3.	**输出结果**：

​	•	如果 MST 的边数达到 V-1，输出其总权值。

​	•	否则，输出 -1 表示图不连通。



**2.3 算法复杂度**

​	•	**时间复杂度**：

​	•	边的排序：$O(E \log E)$。

​	•	并查集操作：每次查找或合并的均摊复杂度为 $O(\alpha(V))$，共进行 E 次操作。

​	•	总时间复杂度：$O(E \log E + E \alpha(V))$。

​	•	**空间复杂度**：$O(V + E)$，存储边集和并查集。



**3.1 代码实现**

```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v, w;
    bool operator<(const Edge& other) const {
        return w < other.w; // 按边权升序排序
    }
};

vector<int> parent, treeRank; // 修改变量名

// 查找操作（路径压缩）
int find(int x) {
    if (x != parent[x]) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

// 合并操作（按秩合并）
bool unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    if (rootX == rootY) return false;
    if (treeRank[rootX] > treeRank[rootY]) { // 使用新的变量名
        parent[rootY] = rootX;
    } else if (treeRank[rootX] < treeRank[rootY]) { // 使用新的变量名
        parent[rootX] = rootY;
    } else {
        parent[rootY] = rootX;
        treeRank[rootX]++; // 使用新的变量名
    }
    return true;
}

int main() {

    int V, E;
    cin >> V >> E;

    vector<Edge> edges(E);
    for (int i = 0; i < E; ++i) {
        cin >> edges[i].u >> edges[i].v >> edges[i].w;
    }

    // Kruskal 初始化
    parent.resize(V + 1);
    treeRank.resize(V + 1, 0); // 使用新的变量名
    for (int i = 1; i <= V; ++i) parent[i] = i;

    // Kruskal 算法
    sort(edges.begin(), edges.end()); // 按边权升序排序
    int mstCost = 0, edgesUsed = 0;

    for (const auto& edge : edges) {
        if (unite(edge.u, edge.v)) { // 如果两点不在同一连通分量，加入 MST
            mstCost += edge.w;
            edgesUsed++;
            if (edgesUsed == V - 1) break; // 如果已经加入 V-1 条边，停止
        }
    }

    // 判断是否连通
    if (edgesUsed != V - 1) {
        cout << -1 << "\n"; // 图不连通
    } else {
        cout << mstCost << "\n";
    }

    return 0;
}
```

**4. 实验结果**

**4.1 测试用例 1**

输入：

```c++
4 5
1 2 2
1 3 2
1 4 3
2 3 4
3 4 3
```

输出：

```c++
7
```

**4.1 测试用例 2**

输入：

```c++
3 2
1 2 5
2 3 7
```

输出：

```c++
12
```

**4.1 测试用例 3**

输入：

```c++
4 2
1 2 5
3 4 7
```

输出：

```c++
-1
```

**实验心得**

**1. Huffman 编码**

通过 Huffman 编码实验，我学习了如何利用小根堆构建最优二叉树，以实现编码长度的最小化。该实验让我深刻理解了贪心策略在构造最优解中的重要性，尤其是在频繁合并操作中，小根堆的高效性尤为显著。此外，编码长度的计算过程也帮助我强化了对加权路径长度的数学模型理解。

**2. 最短路径（Dijkstra 算法）**

通过 Dijkstra 算法实验，我掌握了如何在非负权无向图中高效计算单源最短路径。实验让我体会到贪心策略的精妙，优先队列的使用显著提高了算法在大规模图中的性能。尤其是在连通性判断和距离更新中，我对“最优子结构”和“松弛操作”的概念有了更深刻的认识，也学会了处理不连通情况的边界条件。

**3. 最小生成树 1（Prim 算法）**

通过 Prim 算法实验，我理解了如何从一个初始点出发，逐步扩展生成树，同时最小化边权总和。实验让我认识到优先队列在动态维护边权最小值中的高效性，也让我熟悉了处理稠密图的常用技巧。在测试图不连通的情况时，我进一步强化了对 MST 问题连通性条件的理解，并认识到贪心策略在每一步选择中的局部最优性。

**4. 最小生成树 2（Kruskal 算法）**

通过 Kruskal 算法实验，我深刻体会了“按边排序”的贪心策略在全局优化中的作用，并查集的高效性让我能快速判断边是否会形成环。在实现路径压缩和按秩合并的过程中，我对数据结构的优化策略有了更深入的认识。同时，通过边排序和逐步构建生成树，我体会到 Kruskal 算法在稀疏图中的优势，并进一步理解了其与 Prim 算法在适用场景上的差异。

