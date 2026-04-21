并查集（Union-Find / Disjoint Set Union, DSU）是一种专门用于处理**动态连通性**问题的数据结构。它高效地解决了“分组”与“合并”问题，核心操作接近常数时间，是算法竞赛和工程中不可或缺的“隐士”。

## 一、它到底解决什么问题？

想象这样一个场景：你有一堆点（比如社交网络中的用户），你需要频繁地判断两个点是否属于同一个圈子（连通性），并且需要随时将两个圈子合并。如果用传统的遍历或链表，在大数据量下效率极低。

**并查集的核心能力**：

- **合并 (Union)**：将两个元素所在的集合合并成一个。
- **查找 (Find)**：快速查询某个元素属于哪个集合（通常用“代表元”标识）。
- **判连 (isConnected)**：判断两个元素是否连通。

**典型应用场景**：

- 最小生成树算法（Kruskal 算法）。
- 社交网络中的好友关系判断。
- 编译器中的变量等价性分析。
- 游戏中的像素连通区域标记。

## 二、运作机制与核心优化

并查集通过维护一个**父指针数组**（parent array）来工作。初始时，每个元素都是自己的父节点（自环）。

### 1. 查找 (Find) 与路径压缩

查找操作的目标是找到某个节点的**根节点**（代表元）。最朴素的实现是不断向上递归查找父节点。

**路径压缩优化**：在查找过程中，将路径上的所有节点直接挂载到根节点下。这一步优化能让树的高度急剧降低，后续查找变为接近 O(1) 的操作。

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 递归压缩路径
    return parent[x]
```

### 2. 合并 (Union) 与按秩合并

合并操作不是随意挂载的，否则可能形成长链导致性能退化。

**按秩合并优化**：总将“小树”的根节点挂到“大树”的根节点下（这里的“秩”可以是高度或集合大小）。这能有效控制树的高度增长。

```python
def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry: return  # 已在同一集合
    
    # 按秩合并：小树挂大树
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1  # 高度相同，合并后高度+1
```

## 三、完整 Python 实现

以下是结合了**路径压缩**和**按秩合并**的高效并查集实现。

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # 初始每个节点自成一派
        self.rank = [0] * n           # 记录树的高度（秩）
        self.count = n                # 连通分量个数

    def find(self, x):
        # 路径压缩：递归将路径上所有节点指向根
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return  # 已连通
        
        # 按秩合并：小树挂到大树下
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        self.count -= 1  # 每合并一次，连通分量减1

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)
```

## 四、时间复杂度分析

并查集的时间复杂度分析非常反直觉，它不是一个简单的多项式，而是**阿克曼函数的反函数** α(n)。

| 操作   | 朴素实现 | 仅路径压缩 | 路径压缩 + 按秩合并 |
| ------ | :------: | :--------: | :-----------------: |
| 初始化 |   O(N)   |    O(N)    |        O(N)         |
| Find   |   O(N)   |  O(logN)   |     **O(α(N))**     |
| Union  |   O(N)   |  O(logN)   |     **O(α(N))**     |

**关于 α(n)**：这是一个增长极其缓慢的函数。对于宇宙内的原子数量级（n≈1080），α(n)也不会超过 5。因此，在工程应用和算法竞赛中，我们**近似认为并查集的操作是常数时间复杂度**。

## 五、实战技巧与变种

### 1. 带权并查集

在维护连通性的同时，记录节点与根节点之间的**相对关系**（如距离、差值）。常用于解决“食物链”、“奇偶性”等问题。

- 在 `find`和 `union`中维护一个 `weight`数组，更新相对权重。

### 2. 动态大小处理

如果节点编号不连续或范围未知，可以使用字典（HashMap）代替数组来存储 parent 关系，实现动态扩展。

### 3. 统计连通分量大小

维护一个 `size`数组，在合并时更新根节点的大小信息，即可快速查询任意集合的元素数量。

## 六、总结

并查集是一种“以空间换时间”的典范。它的核心思想是**用父指针表示集合，用路径压缩和按秩合并来保证高效性**。

**核心要点回顾**：

- **用途**：动态连通性问题。
- **核心操作**：`find`（路径压缩）和 `union`（按秩合并）。
- **复杂度**：近似常数时间，效率极高。
- **关键**：理解“代表元”的概念和优化原理。

## 七、题目样例

如题，现在有一个并查集，你需要完成合并和查询操作。

### 1. 输入格式

第一行包含两个整数 $N,M$ ,表示共有$N$个元素和$M$个操作。

接下来$M$行，每行包含三个整数$Z\_i,X\_i,Y\_i$。

1. 当$Z\_i=1$时，将$X\_i$与$Y\_i$所在的集合合并。
2. 当$Z\_i=2$时，输出$X\_i$与$Y\_i$是否在同一集合内，是的输出 `Y`；否则输出`N`。

### 2. 输出格式

对于每一个$Z\_i=2$的操作，都有一行输出，每行包含一个大写字母，为`Y`或者`N`。

### 3. 示例代码

```python
def solve():
    # 读取两个整数N, M
    N, M = map(int, input().strip().split())
    # 创建父节点列表
    parent = [i for i in range(N)]
    
    def find(x):
        if parent[x] != x:
            # 如果当前节点不是父节点，则递归查找
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            parent[x_root] = y_root
    
    # 依次按照输入进行操作
    for _ in range(M):
        Z, X, Y = map(int, input().strip().split())
        
        if Z == 1:
            # 将X, Y所在的集合进行合并
            union(X - 1, Y - 1)

        elif Z == 2:
            # 输出X， Y是否在同一个集合内
            if find(X - 1) == find(Y - 1):
                print('Y')
            else:
                print('N')

if __name__ == '__main__':
    solve()
```

