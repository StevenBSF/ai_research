![image-20241108133233157](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108133233157.png)

## ***计算机学院（国家示范性软件学院）***

# 算法设计与分析实验报告

## 分治与递归

<center>姓名：包诗峰</center>
<center>学号：2022211656</center>
<center>班级：2022211301</center>

<div STYLE="page-break-after: always;"></div> 

## 实验内容

用编程语言实现以下4个算法：

- 归并排序 （迭代实现）

- 快速排序

- 线性时间选择算法

- 平面最接近点对算法

## 实验环境

- C++20

- Clion IDE

- MacOS

  

## 实验步骤

  #### 1.归并排序

  ##### 1.1题目描述

  对 n 个整数使用归并排序进行升序排列排序。

  ##### 1.2 输入格式

  输入文件名为 `mergesort.in`，输入共两行。

  第一行包含一个正整数 n。

  第二行包含 n 个整数 $n_i$，每两个整数之间用空格隔开。

  ##### 1.3 输出格式

  输出文件名为 `mergesort.out`，输出共一行。

  第一行包含 n 个整数 $n_i$，为排序后的升序序列，每两个整数之间用空格隔开。

  ##### 1.4 输入输出样例

  | mergesort.in | mergesort.out |
  | ------------ | ------------- |
  | 5            | 5 9 11 12 22  |
  | 9 11 5 22 12 |               |

  ##### 1.5 数据范围

  $0 < n \le 10^6$

  $|n_i| \le 10^8$ 

  ##### 1.6 补充要求

  改成非递归的方法。

  ##### 1.7 设计思路

  为了实现非递归版本的归并排序，我采用了自底向上的方法，即从最小的子数组开始，逐步扩大合并的范围。这样，每个单元素被视为一个有序的小区间，通过不断地成对合并这些小区间，最终得到一个完整的有序数组。这种方法避免了递归调用的开销，既简化了代码结构，又有效提高了效率。

  ##### 1.8 代码实现

  ```c++
  #include <deque>
  #include <algorithm>
  #include <iostream>
  #include <fstream>
  using namespace std;
  
  #define rep(i, a, b) for (int i = a; i <= b; ++i)
  
  void mergeSortIterative(deque<int>& q) {
      int n = q.size();
      for (int width = 1; width < n; width *= 2) {
          for (int i = 0; i < n; i += 2 * width) {
              int left = i;
              int mid = min(i + width, n);
              int right = min(i + 2 * width, n);
              deque<int> sorted;
              int l = left, r = mid;
              while (l < mid && r < right) {
                  if (q[l] <= q[r]) {
                      sorted.push_back(q[l++]);
                  } else {
                      sorted.push_back(q[r++]);
                  }
              }
              while (l < mid) sorted.push_back(q[l++]);
              while (r < right) sorted.push_back(q[r++]);
              rep(k, 0, sorted.size() - 1) q[left + k] = sorted[k];
          }
      }
  }
  
  
  
  int main(void) {
      ifstream fin("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/quicksort.in");
      if (!fin) {
          cerr << "Error: Input file not found." << endl;
          return 1;
      }
      ofstream fout("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/quicksort.out");
      if (!fout) {
          cerr << "Error: Output file could not be created." << endl;
          return 1;
      }
      int n, tmp;
      deque<int> q;
      fin >> n;
      rep(i, 0, n - 1) {
          fin >> tmp;
          q.push_back(tmp);
      }
  
      mergeSortIterative(q);
      
      rep(i, 0, n - 1) {
          fout << q[i] << (i == n - 1 ? "" : " ");
      }
      return 0;
  }
  
  ```

  ##### 1.9 实现结果

  除了实现样例的输入输出以外，我们考虑如下输入：

  1.9.1 输入：

  ```bash
  15
  200 150 150 300 250 100 50 50 250 300 0 25 75 125 175
  ```

  得到输出结果为：

  ```bash
  0 25 50 50 75 100 125 150 150 175 200 250 250 300 300
  ```

  1.9.2 输入：

  ```bash
  20
  500 400 300 200 100 50 50 75 75 150 150 250 250 325 325 425 425 525 525 600
  ```

  得到输出结果为：

  ```bash
  50 50 75 75 100 150 150 200 250 250 300 325 325 400 425 425 500 525 525 600
  ```

  1.9.3 输入：

  ```bash
  12
  88 12 75 55 66 33 44 22 99 11 0 100
  ```

  得到输出结果为：

  ```bash
  0 11 12 22 33 44 55 66 75 88 99 100
  ```

  ##### 1.10 算法复杂度分析

  对于非递归归并排序的算法复杂度分析，我从时间复杂度和空间复杂度两个方面来探讨。

  时间复杂度方面：

  非递归归并排序的时间复杂度仍然是 O(nlogn)。在这个算法中，我们通过逐步增大子数组的宽度进行合并，每一轮的合并操作都需要遍历整个数组。因为每次宽度 `width` 都会翻倍，整个数组合并的轮数为 log⁡n次，而每轮都需要 O(n) 的时间。因此，总体的时间复杂度为 O(nlogn)，与递归版本的归并排序相同。

   空间复杂度方面：

  在空间复杂度方面，这个算法使用了一个额外的 `deque<int>` 来存储临时合并结果，所以每次合并都需要 O(n) 的额外空间。由于每一轮的合并过程都需要新的临时存储空间，整个算法的空间复杂度也是 O(n)。

  综上，非递归归并排序的时间复杂度和递归版本保持一致，都是 O(nlog⁡n)，但同样需要额外的 O(n)空间来临时存储数据，确保排序过程的稳定性。这种非递归的实现避免了递归带来的函数栈开销，在某些场景下可能表现得更高效。

  

  #### 2. 快速排序

  ##### 2.1 题目描述

  对 n 个整数使用快速排序进行升序排列。

  ##### 2.2 输入格式

  输入文件名为 `quicksort.in`。输入共两行。

  - 第一行包含一个正整数 n。
  - 第二行包含 n 个整数 $n_i$，每两个整数之间用空格隔开。

  ##### 2.3 输出格式

  输出文件名为 `quicksort.out`。输出共一行。

  - 第一行包含 n 个整数 $n_i$，为排序后的升序序列，每两个整数之间用空格隔开。

  ##### 2.4 输入输出样例

  | quicksort.in | quicksort.out |
  | ------------ | ------------- |
  | 5            | 5 9 11 12 22  |
  | 9 11 5 22 12 |               |

  ##### 2.5 数据范围

  - $0 < n \le 2 \times 10^6$
  - $|n_i| \le 10^8$

##### 2.6 设计思路

​	![image-20241108141634201](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108141634201.png)

考虑对于ppt中对于算法设计的改进，我依照这个三步骤进行代码编写。

##### 2.7 代码实现

```c++
#include <deque>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

#define rep(i, a, b) for (int i = a; i <= b; ++i)

void quickSort(deque<int>& q, int left, int right) {
    // Step 1:
    bool nonDecreasing = true;
    for (int i = left; i < right; ++i) {
        if (q[i] > q[i + 1]) {
            nonDecreasing = false;
            break;
        }
    }
    if (nonDecreasing) {
        return;
    }

    // Step 2:
    bool nonIncreasing = true;
    for (int i = left; i < right; ++i) {
        if (q[i] < q[i + 1]) {
            nonIncreasing = false;
            break;
        }
    }
    if (nonIncreasing) {
        // 逆序调整为升序
        reverse(q.begin() + left, q.begin() + right + 1);
        return;
    }

    // Step 3:
    if (left < right) {
        int pivot = q[right];
        int i = left - 1;
        for (int j = left; j < right; ++j) {
            if (q[j] <= pivot) {
                ++i;
                swap(q[i], q[j]);
            }
        }
        swap(q[i + 1], q[right]);
        int partitionIndex = i + 1;
        quickSort(q, left, partitionIndex - 1);
        quickSort(q, partitionIndex + 1, right);
    }
}

int main(void) {
    ifstream fin("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/quicksort.in");
    if (!fin) {
        cerr << "Error: Input file not found." << endl;
        return 1;
    }
    ofstream fout("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/quicksort.out");
    if (!fout) {
        cerr << "Error: Output file could not be created." << endl;
        return 1;
    }
    int n, tmp;
    deque<int> q;
    fin >> n;
    rep(i, 0, n - 1) {
        fin >> tmp;
        q.push_back(tmp);
    }

    quickSort(q, 0, n - 1);
    rep(i, 0, n - 1) {
        fout << q[i] << (i == n - 1 ? "" : " ");
    }
    return 0;
}

```

##### 2.9 实现结果

除了实现样例的输入输出以外，我们考虑如下输入：

2.9.1 输入：

```bash
15
200 150 150 300 250 100 50 50 250 300 0 25 75 125 175
```

得到输出结果为：

```bash
0 25 50 50 75 100 125 150 150 175 200 250 250 300 300
```

2.9.2 输入：

```bash
20
500 400 300 200 100 50 50 75 75 150 150 250 250 325 325 425 425 525 525 600
```

得到输出结果为：

```bash
50 50 75 75 100 150 150 200 250 250 300 325 325 400 425 425 500 525 525 600
```

2.9.3 输入：

```bash
12
88 12 75 55 66 33 44 22 99 11 0 100
```

得到输出结果为：

```bash
0 11 12 22 33 44 55 66 75 88 99 100
```



##### 2.10 复杂度分析

对于快速排序的算法复杂度分析，同样可以从时间复杂度和空间复杂度两方面来分析。

快速排序的时间复杂度依赖于划分过程的效率，即每次选取的“枢轴”能否将数组大致等分。

- **平均情况**：在理想情况下，快速排序的每一次划分都将数组分成两半，因此时间复杂度为 O(nlog⁡n)。这是因为，每次划分需要 O(n)的操作，而划分的层数为 logn，总的时间复杂度为 O(nlogn)。这也是快速排序的典型表现。
- **最坏情况**：在最坏情况下，比如数组已经有序且选择了最左边或最右边作为枢轴，每次划分都只划分出一个小区间，导致划分层数达到O(n)。在这种情况下，快速排序的时间复杂度退化为 O(n2)。
- **最优情况**：快速排序的最优表现和平均情况一致，都是 O(nlogn)

快速排序的空间复杂度主要取决于递归的层数：

- **平均情况**：在理想条件下，递归深度为 logn，因此平均空间复杂度为O(logn)。
- **最坏情况**：如果递归深度达到 O(n)（例如最坏情况时的有序数组），则空间复杂度会退化为 O(n)。



#### 3. 线性时间选择

##### 3.1 题目描述

在给定线性序列的 n 个元素中找出第 k 小的元素。

##### 3.2 输入格式

输入文件名为 `select.in`，输入共两行。

- 第一行包含两个正整数 n, k，两个数之间用空格隔开。
- 第二行包含 n 个整数 $n_i$，每两个整数之间用空格隔开。

##### 3.3 输出格式

输出文件名为 `select.out`，输出共一行。

- 第一行包含一个整数，表示第 k 小的元素。

##### 3.4 输入输出样例

| select.in    | select.out |
| ------------ | ---------- |
| 5 2          | 9          |
| 9 11 5 22 12 |            |

##### 3.5 数据范围

- $0 < n \le 8 \times 10^5$
- $1 \le k \le n$
- $|n_i| \le 10^8$

##### 3.6 说明/提示

要求使用***一分为三***的减治法。

##### 3.7 设计思路

![image-20241108143733909](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108143733909.png)

参考老师在ppt中的设计思路，我们将区间划分为三部分。

下面是我的设计思路：

在一些应用场景中，我们需要在无序数组中找到第 `k` 小的元素（比如中位数或其他特定顺位的元素），但我们并不需要对整个数组进行排序。排序的时间复杂度是O(nlogn)，而我们只需要找到一个特定位置的元素。快速选择算法通过部分排序，将时间复杂度降到了 O(n)O(n)，这是我们设计的主要出发点。

这里我们使用了一种减治法来解决问题，即逐步缩小待查找的范围，直到找到目标位置的元素。为了加快收敛速度，我们采用“三分法”的变种，将区间分为三部分，进一步优化选择过程。

`partition` 函数的作用是将选定的枢轴（pivot）元素放到它最终的位置上，且确保比它小的元素在左边，比它大的元素在右边。具体步骤如下：

- 将枢轴元素暂时移到右边。
- 遍历数组，将小于枢轴的元素移动到 `storeIndex` 位置。
- 遍历结束后，把枢轴元素放回 `storeIndex`，并返回 `storeIndex`，即枢轴的最终位置。

这样，枢轴元素就在它最终的排序位置上，左右两侧的元素各自小于或大于它，从而为后续步骤缩小了范围。

`select` 函数是主算法，用于递归地缩小查找范围。在每一步中，将数组三等分，分别确定两个中间位置 `mid1`和 `mid2`，根据 `k` 所在的位置选择枢轴。算法过程是这样的：

- 如果 `k` 位于左区间，就将 `mid1` 作为枢轴。
- 如果 `k` 位于中间区间，就选择 `mid2` 作为枢轴。
- 如果 `k` 位于右区间，选择最右侧元素作为枢轴。

选择枢轴后，通过 `partition` 函数对数组进行分区，并根据 `k` 的位置判断下一步是否缩小左边界或右边界，直到找到目标位置的元素。



##### 3.8 代码实现

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

int partition(vector<int>& arr, int left, int right, int pivotIndex) {
    int pivotValue = arr[pivotIndex];
    swap(arr[pivotIndex], arr[right]);
    int storeIndex = left;
    for (int i = left; i < right; ++i) {
        if (arr[i] < pivotValue) {
            swap(arr[i], arr[storeIndex]);
            ++storeIndex;
        }
    }
    swap(arr[storeIndex], arr[right]);
    return storeIndex;
}

int select(vector<int>& arr, int left, int right, int k) {
    while (left <= right) {
        if (left == right) {
            return arr[left];
        }
        // 一分为三的减治法
        int mid1 = left + (right - left) / 3;
        int mid2 = left + 2 * (right - left) / 3;

        int pivotIndex;
        if (k <= mid1) {
            pivotIndex = mid1;
        } else if (k > mid1 && k <= mid2) {
            pivotIndex = mid2;
        } else {
            pivotIndex = right;
        }
        pivotIndex = partition(arr, left, right, pivotIndex);
        if (k == pivotIndex) {
            return arr[k];
        } else if (k < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }
    return -1; // Should never reach here if k is valid
}

int main() {
    ifstream fin("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/select.in");
    if (!fin) {
        cerr << "Error: Input file not found." << endl;
        return 1;
    }
    ofstream fout("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/select.out");
    if (!fout) {
        cerr << "Error: Output file could not be created." << endl;
        return 1;
    }
    int n, k;
    fin >> n >> k;
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        fin >> arr[i];
    }
    int result = select(arr, 0, n - 1, k - 1);
    fout << result << endl;
    return 0;
}

```



##### 3.9 实现结果

除了实现样例的输入输出以外，我们考虑如下输入：

3.9.1 输入：

```bash
10 5
15 20 10 30 25 40 5 35 45 50
```

得到输出结果为：

```bash
25
```

3.9.2 输入：

```bash
8 6
4 2 5 2 8 9 3 2
```

得到输出结果为：

```bash
4
```

3.9.3 输入：

```bash
6 6
3 1 4 1 5 9
```

得到输出结果为：

```bash
9
```



#### 4 平面最近点对

##### 4.1 题目描述

给出 n 个二维平面上的点，求一组欧几里得距离最近的点对。

##### 4.2 输入格式

输入文件名为 `point.in`，输入共 n+1 行。

- 第一行包含一个正整数 n。
- 接下来 n 行每行包含 2 个整数 $x_i$, $y_i$，两数之间用空格隔开。

##### 4.3 输出格式

输出文件名为 `point.out`，输出共一行。

- 第一行包含一个小数，表示最小点对间的距离，保留两位小数。

##### 4.4 输入输出样例

| point.in | point.out |
| -------- | --------- |
| 5        | 1.00      |
| 0 0      |           |
| 2 0      |           |
| 0 1      |           |
| 2 2      |           |
| 1 1      |           |

##### 4.5 数据范围

- $2 \le n \le 7 \times 10^5$
- $|x_i| \le 10^6, |y_i| \le 10^6$

##### 4.6 说明/提示

输入数据保证没有任意两点是重合的。

![image-20241108145106642](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108145106642.png)

##### 4.7 设计思路

参照PPT，按照PPT中的六个步骤进行编写代码。

##### 4.8 代码实现

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace std;

struct Point {
    int x, y;
};

bool compareX(const Point& p1, const Point& p2) {
    return p1.x < p2.x;
}

bool compareY(const Point& p1, const Point& p2) {
    return p1.y < p2.y;
}

double distance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

pair<Point, Point> closestPairUtil(vector<Point>& points, int left, int right) {
    if (right - left <= 3) {
        double minDist = numeric_limits<double>::infinity();
        pair<Point, Point> closestPair;
        for (int i = left; i < right; ++i) {
            for (int j = i + 1; j < right; ++j) {
                double dist = distance(points[i], points[j]);
                if (dist < minDist) {
                    minDist = dist;
                    closestPair = {points[i], points[j]};
                }
            }
        }
        return closestPair;
    }

    int mid = left + (right - left) / 2;
    Point midPoint = points[mid];

    // Step 1: 找中垂线 l
    pair<Point, Point> leftPair = closestPairUtil(points, left, mid);
    pair<Point, Point> rightPair = closestPairUtil(points, mid, right);
    double dl = distance(leftPair.first, leftPair.second);
    double dr = distance(rightPair.first, rightPair.second);
    double dm = min(dl, dr);
    pair<Point, Point> closestPair = (dl < dr) ? leftPair : rightPair;

    // Step 2: 筛选中垂线 P1, P2 范围内的点
    vector<Point> strip;
    for (int i = left; i < right; ++i) {
        if (abs(points[i].x - midPoint.x) < dm) {
            strip.push_back(points[i]);
        }
    }

    // Step 3: 在 P1, P2 范围内，按照 y 坐标排序，从下到上依次判断最近点对
    sort(strip.begin(), strip.end(), compareY);
    double minDist = dm;
    for (int i = 0; i < strip.size(); ++i) {
        for (int j = i + 1; j < strip.size() && (strip[j].y - strip[i].y) < minDist; ++j) {
            double dist = distance(strip[i], strip[j]);
            if (dist < minDist) {
                minDist = dist;
                closestPair = {strip[i], strip[j]};
            }
        }
    }

    // Step 4: 将 P1、P2 中的点按 x 坐标升序排序，形成点列 X
    sort(strip.begin(), strip.end(), compareX);

    // Step 5: 依次扫描 X 中各点，对于每个点，检查 Y 中与其距离在 dm 之内的所有点（最多 6 个）
    vector<Point> yStrip = strip; // Y 点列
    sort(yStrip.begin(), yStrip.end(), compareY);
    for (int i = 0; i < strip.size(); ++i) {
        for (int j = i + 1; j < yStrip.size() && (yStrip[j].y - yStrip[i].y) < dm; ++j) {
            double dist = distance(yStrip[i], yStrip[j]);
            if (dist < minDist) {
                minDist = dist;
                closestPair = {yStrip[i], yStrip[j]};
            }
        }
    }

    // Step 6: 计算最终最小距离 d = min(dm, dl)，并返回最近的点对和最小距离
    double d = min(dm, minDist);
    if (d == minDist) {
        return closestPair;
    } else if (d == dl) {
        return leftPair;
    } else {
        return rightPair;
    }
}

pair<Point, Point> closestPair(vector<Point>& points) {
    sort(points.begin(), points.end(), compareX);
    return closestPairUtil(points, 0, points.size());
}

int main() {
    ifstream fin("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/point.in");
    if (!fin) {
        cerr << "Error: Input file not found." << endl;
        return 1;
    }
    ofstream fout("/Users/baoshifeng/Desktop/homework/semester5/algorithm/lab1/point.out");
    if (!fout) {
        cerr << "Error: Output file could not be created." << endl;
        return 1;
    }
    int n;
    fin >> n;
    vector<Point> points(n);
    for (int i = 0; i < n; ++i) {
        fin >> points[i].x >> points[i].y;
    }
    pair<Point, Point> resultPair = closestPair(points);
    double resultDistance = distance(resultPair.first, resultPair.second);
    fout << "Closest pair: (" << resultPair.first.x << ", " << resultPair.first.y << ") and (" << resultPair.second.x << ", " << resultPair.second.y << ")" << endl;
    fout << "Distance: " << fixed << setprecision(2) << resultDistance << endl;
    return 0;
}

```



##### 4.9 实现结果

本例由于修改了输出要求，所以相应的输出也有一定改变。对于样例输入：

```bash
5
0 0
2 0
0 1
2 2
1 1
```

得到输出结果为：

```bash
Closest pair: (0, 0) and (0, 1)
Distance: 1.00
```

## 实验心得

对于前两个实验排序部分，由于大一参加过程序设计竞赛所以相对比较熟悉，对于第四个题目由于按照ppt设计思路设计耗费了一定的时间。收获很大。





