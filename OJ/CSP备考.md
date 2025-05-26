题单

https://www.acwing.com/blog/content/9857/





![image-20250525010956872](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20250525010956872.png)

![image-20250525011010310](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20250525011010310.png)



当然可以，下面是一个完整的 **C++ 对拍流程示例**，包括：

- 主程序（高效解）
- 暴力程序（慢但正确）
- 数据生成器（生成随机测试数据）
- 对拍脚本（Python 编写）



## **🌟 题目示例：给定长度为** n **的数组，求两个数之和为定值** k **的方案数。**

## **一、主程序（main.cpp） — 快速解法** O(n \log n)

```
// main.cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    unordered_map<int, int> count;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        count[a[i]]++;
    }
    long long ans = 0;
    for (auto [x, c] : count) {
        int y = k - x;
        if (count.count(y)) {
            if (x == y)
                ans += 1LL * c * (c - 1) / 2;
            else if (x < y)
                ans += 1LL * c * count[y];
        }
    }
    cout << ans << endl;
    return 0;
}
```

## **二、暴力程序（brute.cpp）— 暴力枚举** O(n^2)

```
// brute.cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; ++i)
        cin >> a[i];
    int ans = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (a[i] + a[j] == k)
                ++ans;
    cout << ans << endl;
    return 0;
}
```

## **三、数据生成器（gen.cpp）**



```
// gen.cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n = rand() % 20 + 1; // 数组长度：1~20
    int k = rand() % 40;
    cout << n << " " << k << endl;
    for (int i = 0; i < n; ++i)
        cout << rand() % 20 << " ";
    cout << endl;
    return 0;
}
```

编译以上三个程序：

```
g++ main.cpp -o main
g++ brute.cpp -o brute
g++ gen.cpp -o gen
```

## **四、对拍脚本（test.py）**

```
import os

for t in range(1000):
    os.system("./gen > data.in")
    os.system("./main < data.in > out1.txt")
    os.system("./brute < data.in > out2.txt")
    
    with open("out1.txt") as f1, open("out2.txt") as f2:
        ans1 = f1.read().strip()
        ans2 = f2.read().strip()
        if ans1 != ans2:
            print(f"Wrong Answer on Test #{t}")
            with open("data.in") as fin:
                print("Input:")
                print(fin.read())
            print("Your output:", ans1)
            print("Expected:", ans2)
            break
else:
    print("All tests passed!")
```

## **✅ 总结**

这就是对拍的完整流程：

1. main.cpp 是你的算法；
2. brute.cpp 是保证正确的暴力方法；
3. gen.cpp 自动生成输入数据；
4. test.py 比较两者输出，一旦不一致，立刻暴露问题！

如你想要这个流程用于某个具体题目，也可以告诉我题目内容，我可以帮你定制生成器和暴力代码。



