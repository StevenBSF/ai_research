![image-20241108133233157](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108133233157.png)

## ***计算机学院（国家示范性软件学院）***

# 算法设计与分析实验报告

## 分治与递归

<center>姓名：包诗峰</center>
<center>学号：2022211656</center>
<center>班级：2022211301</center>

<div STYLE="page-break-after: always;"></div> 



## 实验环境

- C++20

- Clion IDE

- MacOS

#### 1. 实验内容

##### 1.1 题目描述

给定一个背包容量为 $V$ 的背包和 $N$ 个物品，每个物品有重量 $w_i$ 和价值 $v_i$。要求在不超过背包容量的前提下，选择物品使得总价值最大。

##### 1.2 输入格式

输入文件名为 `bag.in`，输入共两行。

- 第一行包含两个整数 $V$ 和 $N$，分别表示背包容量和物品数量。
- 第二行包含 $N$ 个物品的重量和价值，每两个数之间用空格隔开，格式为 $w_1, v_1, w_2, v_2, \ldots$。

##### 1.3 输出格式

输出文件名为 `bag.out`，输出共一行。

第一行包含一个整数，为最大价值。

第二行为最优组合

##### 1.4 输入输出样例

| bag.in | bag.out |
| ------ | ------- |
| 70 3   | 3       |
| 71 100 | 2 3     |
| 69 1   |         |
| 1 2    |         |

##### 1.5 数据范围

- $ 0 < N \leq 2500 $
- $ 0 < V \leq 2500 $
- $ 0 < c_i \leq 2500 $
- $ 0 < w_i \leq 1000000 $

------

#### 2. 实验设计思路



在0-1背包问题中，给定一个背包的容量 $V$ 和 $N$ 件物品，每件物品有一个重量 $w[i]$ 和价值$v[i]$，目标是选择一些物品放入背包，使得背包的总价值最大，并且物品的总重量不超过背包容量$V$。

在基础版本中，使用二维数组 `dp[i][j]` 来表示前 $ i $ 件物品放入容量为 $ j $ 的背包时的最大价值。

##### 2.1 状态定义

- `dp[i][j]`：表示考虑前 $ i $ 件物品，在背包容量为 $ j $ 时，能够达到的最大价值。

##### 2.2 状态转移方程

对于每一件物品 $ i $，有两种选择：
1. **不放入背包**：如果不放入背包，则最大价值为前$ i-1 $ 件物品在容量 $ j $ 下的最大价值，即：`dp[i][j] = dp[i-1][j]`。
2. **放入背包**：如果将物品 $ i $ 放入背包，并且背包容量 $ j $ 大于等于物品 $ i $ 的重量 $ w[i] $，则可以得到最大价值为前 $ i-1$ 件物品在容量 $ j-w[i] $ 下的最大价值，加上物品 $ i $ 的价值 $ v[i] $，即：`dp[i][j] = dp[i-1][j-w[i]] + v[i]`。

根据上述两种情况，动态规划的状态转移方程为：
$$dp[i][j] = \max(dp[i-1][j], dp[i-1][j - w[i]] + v[i]) \quad \text{当} \quad j \geq w[i]$$
当背包容量 $ j $ 小于当前物品$ i $ 的重量$ w[i] $ 时，不能放入该物品，所以：
$$dp[i][j] = dp[i-1][j] \quad \text{,} \quad j < w[i]$$

##### 2.3 边界条件

- `dp[0][j] = 0`：当没有物品时，不管背包容量多大，最大价值都是 0。
- `dp[i][0] = 0`：当背包容量为 0 时，放不下任何物品，最大价值也是 0。

**基础版本动态规划公式：**

对于给定的物品 $ i $和背包容量 $ j $，选择是否放入物品 $ i $ 的动态规划公式为：
$$dp[i][j] = \max(dp[i-1][j], dp[i-1][j - w[i]] + v[i])$$
对于物品 $ i $ 和背包容量 $ j $，有以下两种情况：

- **不放入物品 $ i $**：即保持背包容量为 $ j $ 时的最大价值 `dp[i-1][j]`。
- **放入物品 $ i $**：即从背包容量  $j $ 减去物品 $ i $ 的重量 $ w[i] $，并将物品 $ i $ 的价值 $ v[i] $ 加上，得到新的最大价值 `dp[i-1][j-w[i]] + v[i]`。

------

#### 3. 基础算法实现

##### 3.1 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 定义最大容量和物品数量范围
const int MAX_N = 2500;
const int MAX_V = 2500;

// dp[i][j] 表示前 i 件物品放入容量为 j 的背包的最大价值
int dp[MAX_N + 1][MAX_V + 1];

// w[i] 和 v[i] 分别表示第 i 件物品的重量和价值
int w[MAX_N + 1], v[MAX_N + 1];

int main() {
    int V, N; // 背包容量 V 和物品数量 N
    cin >> V >> N;

    // 输入每件物品的重量和价值
    for (int i = 1; i <= N; i++) {
        cin >> w[i] >> v[i];
    }

    // 动态规划求解最大价值
    for (int i = 1; i <= N; i++) {
        for (int j = 0; j <= V; j++) {
            if (j < w[i]) {
                dp[i][j] = dp[i - 1][j]; // 无法放入当前物品
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
            }
        }
    }

    // 输出最大价值
    cout << dp[N][V] << endl;

    return 0;
}
```

##### 3.2 算法复杂度分析

- **时间复杂度**：每个物品在背包容量范围内进行转移计算，总共需要 $O(N \times V)$。
- **空间复杂度**：使用二维数组存储状态，总共需要 $O(N \times V)$​。

此外，为了实现求解最优组合，我们进一步将代码修改如下：
```c++
#include <bits/stdc++.h>
using namespace std;

// 定义最大容量和物品数量范围
const int MAX_N = 2500;
const int MAX_V = 2500;

// dp[i][j] 表示前 i 件物品放入容量为 j 的背包的最大价值
int dp[MAX_N + 1][MAX_V + 1];

// w[i] 和 v[i] 分别表示第 i 件物品的重量和价值
int w[MAX_N + 1], v[MAX_N + 1];

// 记录是否选择了某件物品
bool selected[MAX_N + 1][MAX_V + 1];

int main() {
	int V, N; // 背包容量 V 和物品数量 N
	cin >> V >> N;
	
	// 输入每件物品的重量和价值
	for (int i = 1; i <= N; i++) {
		cin >> w[i] >> v[i];
	}
	
	// 动态规划求解最大价值
	for (int i = 1; i <= N; i++) {
		for (int j = 0; j <= V; j++) {
			if (j < w[i]) {
				dp[i][j] = dp[i - 1][j]; // 无法放入当前物品
			} else {
				if (dp[i - 1][j] > dp[i - 1][j - w[i]] + v[i]) {
					dp[i][j] = dp[i - 1][j];
					selected[i][j] = false; // 当前物品未被选择
				} else {
					dp[i][j] = dp[i - 1][j - w[i]] + v[i];
					selected[i][j] = true; // 当前物品被选择
				}
			}
		}
	}
	
	// 输出最大价值
	cout << dp[N][V] << endl;
	
	// 回溯找出最优组合
	vector<int> chosen_items; // 存储被选择的物品编号
	int capacity = V;
	for (int i = N; i >= 1; i--) {
		if (selected[i][capacity]) { // 如果选择了第 i 件物品
			chosen_items.push_back(i);
			capacity -= w[i]; // 减少剩余容量
		}
	}
	
	// 输出最优组合
	for (int i = chosen_items.size() - 1; i >= 0; i--) { // 从后往前输出
		cout << chosen_items[i] << (i == 0 ? "" : " ");
	}
	cout << endl;
	
	return 0;
}

```



------

#### 4. 改进算法实现

##### 4.1 设计思路

在改进版本中，通过将 `dp[i][j]` 数组优化为一维数组 `f[j]`，进一步优化了空间复杂度，减少了存储的空间。

##### 4.1 状态定义

- `f[j]`：表示背包容量为 $ j $ 时的最大价值。

##### 4.2 状态转移方程

对于每一件物品 $ i $，从背包容量 $ j = V $ 到 $ j = w[i] $ 逆序遍历，更新当前容量下的最大价值：
$$f[j] = \max(f[j], f[j - w[i]] + v[i]) \quad \text{,} \quad j \geq w[i]$$
逆序遍历的原因是为了避免在同一轮更新中重复使用当前物品。由于我们只依赖于上一轮状态，所以采用逆序遍历确保在更新 `f[j]` 时，`f[j-w[i]]` 仍然是上一轮的值，而不是当前轮的值。

##### 4.3 边界条件

- 初始状态：`f[0] = 0`，背包容量为 0 时，最大价值为 0。
- 对于 $f[j] $（$ j > 0 $），在循环开始时它的值是 0，表示没有物品放入背包时的初始值。

**改进版本动态规划公式：**

对于改进版本，空间复杂度的优化通过将 `dp[i][j]` 降为一维数组 `f[j]` 来实现。我们仍然根据是否放入物品 $ i $ 来更新背包容量 $ j $ 时的最大价值：
$$f[j] = \max(f[j], f[j - w[i]] + v[i])$$
此时，逆序遍历确保每件物品在更新状态时只会使用上一轮的值，从而避免了重复使用物品的问题。

##### 4.2 代码实现

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAX_V = 2500; // 背包最大容量
int f[MAX_V + 1]; // f[j] 表示容量为 j 时的最大价值
int w[2501], v[2501]; // 物品重量和价值

/*
  函数功能：求解 0-1 背包问题的最大价值
  函数形参：物品数量 n 和背包容量 m
  函数返回值：返回最大价值
 */
int fun(int n, int m) {
    // 遍历每件物品
    for (int i = 1; i <= n; i++) {
        // 逆序遍历容量，避免重复使用物品
        for (int j = m; j >= w[i]; j--) {
            f[j] = max(f[j], f[j - w[i]] + v[i]);
        }
    }
    return f[m];
}

int main() {
    int V, N; // 背包容量和物品数量
    cin >> V >> N;

    // 输入物品的重量和价值
    for (int i = 1; i <= N; i++) {
        cin >> w[i] >> v[i];
    }

    // 输出最大价值
    cout << fun(N, V) << endl;

    return 0;
}
```

##### 4.3 算法复杂度分析

- **时间复杂度**：$O(N \times V)$，与基础版本一致。
- **空间复杂度**：使用一维数组存储状态，总共需要 $O(V)$​。



为了求出最优组合，我们进一步得到如下版本：

```c++
#include <bits/stdc++.h>
using namespace std;

const int MAX_V = 2500; // 背包最大容量
int f[MAX_V + 1];       // f[j] 表示容量为 j 时的最大价值
int w[2501], v[2501];   // 物品重量和价值
bool choose[2501][MAX_V + 1]; // choose[i][j] 表示是否选择物品 i 达到容量 j 时的最大价值

/*
  函数功能：求解 0-1 背包问题的最大价值和选择的物品
  函数形参：物品数量 n 和背包容量 m
  函数返回值：返回最大价值
 */
int fun(int n, int m) {
	// 遍历每件物品
	for (int i = 1; i <= n; i++) {
		// 逆序遍历容量，避免重复使用物品
		for (int j = m; j >= w[i]; j--) {
			if (f[j - w[i]] + v[i] > f[j]) {
				f[j] = f[j - w[i]] + v[i];
				choose[i][j] = true; // 标记选择了物品 i
			} else {
				choose[i][j] = false;
			}
		}
	}
	return f[m];
}

/*
  函数功能：回溯最优组合
  函数形参：物品数量 n 和背包容量 m
  函数返回值：返回一个包含选中物品索引的列表
 */
vector<int> getItems(int n, int m) {
	vector<int> result;
	int capacity = m;
	
	for (int i = n; i >= 1; i--) {
		if (choose[i][capacity]) {
			result.push_back(i); // 记录选中的物品
			capacity -= w[i];   // 减少背包剩余容量
		}
	}
	
	return result;
}

int main() {
	int V, N; // 背包容量和物品数量
	cin >> V >> N;
	
	// 输入物品的重量和价值
	for (int i = 1; i <= N; i++) {
		cin >> w[i] >> v[i];
	}
	
	// 求解最大价值
	int maxValue = fun(N, V);
	cout << maxValue << endl;
	
	// 求解最优组合
	vector<int> items = getItems(N, V);
	for (int i = items.size() - 1; i >= 0; i--) { // 按输入顺序输出
		cout << items[i] << " ";
	}
	cout << endl;
	
	return 0;
}

```



------

#### 5. 实现结果

对于以下输入，实验结果如下：

##### 5.1 输入：

```bash
50 3
10 60
20 100
30 120
```

##### 输出：

```bash
220
2 3
```

##### 5.2 输入：

```bash
100 5
20 40
50 100
30 60
40 90
10 30
```

##### 输出：

```bash
230
1 2 4
```

##### 5.3 输入：

```bash
100 10
10 20
20 30
30 66
40 90
50 100
60 120
70 130
80 140
90 150
100 160
```

##### 输出：

```bash
210
4 6
```



