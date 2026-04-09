## 一、背包问题概述

背包问题（Knapsack Problem）是组合优化中的一个经典问题：给定一组物品，每种物品都有自己的重量和价值，在限定的总重量内，如何选择物品使得总价值最大。

## 二、背包问题的分类

### 1. 0-1背包问题

- 每种物品只有一件，要么放入（1）要么不放入（0）
- 这是最经典的背包问题

### 2. 完全背包问题

- 每种物品有无限件可用

### 3. 多重背包问题

- 每种物品有有限件可用

### 4. 分组背包问题

- 物品被划分为若干组，每组内的物品互斥

## 三、0-1背包问题的动态规划解法

### 动态规划思路

- 定义`dp[i][w]`：考虑前`i`个物品，背包容量为`w`时的最大价值
- 比较下面两个状态转移方程，取两者的最大值： 
  - 不选第`i`个物品：`dp[i][w] = dp[i-1][w]` 
  - 选第i个物品：`dp[i][w] = dp[i-1][w-weight[i-1]] + value[i-1] `

### 关键表格：DP表

**核心思想**：把大问题分解成小问题，记住已经算过的结果

假设：

- 物品：A(2kg, 3元), B(3kg, 4元)
- 背包容量：5kg

建一个表格，**行表示物品，列表示背包容量**：

| 容量\物品 | 0kg  | 1kg  | 2kg  | 3kg  | 4kg  | 5kg  |
| --------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 没有物品  | 0    | 0    | 0    | 0    | 0    | 0    |
| 只有A     | 0    | 0    | 3    | 3    | 3    | 3    |
| 有A和B    | 0    | 0    | 3    | 4    | 4    | 7    |

## 四、Python代码实现

```python
def knapsack_01(weights, values, capacity):
    """
    0-1背包问题的动态规划解法
    
    参数:
    weights: 物品重量列表
    values: 物品价值列表
    capacity: 背包容量
    
    返回:
    最大价值和选择的物品索引
    """
    n = len(weights)
    
    # 创建动态规划表：dp[i][w] 表示考虑前i个物品，背包容量为w时的最大价值
    # 多一行一列是为了方便处理边界情况
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充dp表
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                # 可以放入当前物品
                dp[i][w] = max(
                    dp[i-1][w],  # 不选当前物品
                    dp[i-1][w-weights[i-1]] + values[i-1]  # 选当前物品
                )
            else:
                # 当前物品太重，无法放入
                dp[i][w] = dp[i-1][w]
    
    # 回溯找出选择的物品
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)  # 物品索引
            w -= weights[i-1]
    
    selected_items.reverse()  # 保持原始顺序
    
    return dp[n][capacity], selected_items


def knapsack_01_optimized(weights, values, capacity):
    """
    0-1背包问题的空间优化解法（一维数组）
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 从后向前遍历，确保每个物品只被考虑一次
        for w in range(capacity, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    
    return dp[capacity]


def complete_knapsack(weights, values, capacity):
    """
    完全背包问题的动态规划解法
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 从前向后遍历，允许重复选择
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    
    return dp[capacity]


def fractional_knapsack_greedy(weights, values, capacity):
    """
    分数背包问题的贪心解法
    可以取物品的一部分
    """
    n = len(weights)
    items = []
    
    # 计算单位价值
    for i in range(n):
        value_per_weight = values[i] / weights[i]
        items.append((value_per_weight, weights[i], values[i], i))
    
    # 按单位价值降序排序
    items.sort(reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    selected_items = []
    
    for vpw, weight, value, idx in items:
        if remaining_capacity >= weight:
            # 可以完整放入
            total_value += value
            remaining_capacity -= weight
            selected_items.append((idx, 1.0))  # 完整选择
        else:
            # 只能放入一部分
            fraction = remaining_capacity / weight
            total_value += value * fraction
            selected_items.append((idx, fraction))
            break  # 背包已满
    
    return total_value, selected_items


def multiple_knapsack(weights, values, counts, capacity):
    """
    多重背包问题的动态规划解法
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # 将多重背包转化为多个0-1背包
        k = 1
        while k < counts[i]:
            # 二进制拆分优化
            weight_k = weights[i] * k
            value_k = values[i] * k
            
            for w in range(capacity, weight_k-1, -1):
                dp[w] = max(dp[w], dp[w-weight_k] + value_k)
            
            counts[i] -= k
            k *= 2
        
        # 处理剩余部分
        if counts[i] > 0:
            weight_k = weights[i] * counts[i]
            value_k = values[i] * counts[i]
            
            for w in range(capacity, weight_k-1, -1):
                dp[w] = max(dp[w], dp[w-weight_k] + value_k)
    
    return dp[capacity]


class KnapsackSolver:
    """背包问题求解器类"""
    
    def __init__(self, weights, values, names=None):
        self.weights = weights
        self.values = values
        self.names = names if names else [f"物品{i}" for i in range(len(weights))]
    
    def solve_01_knapsack(self, capacity):
        """解决0-1背包问题"""
        max_value, selected_indices = knapsack_01(
            self.weights, self.values, capacity
        )
        
        selected_items = []
        total_weight = 0
        for idx in selected_indices:
            selected_items.append({
                'name': self.names[idx],
                'weight': self.weights[idx],
                'value': self.values[idx]
            })
            total_weight += self.weights[idx]
        
        return {
            'max_value': max_value,
            'total_weight': total_weight,
            'selected_items': selected_items,
            'selected_indices': selected_indices
        }
    
    def solve_fractional_knapsack(self, capacity):
        """解决分数背包问题"""
        max_value, selected_fractions = fractional_knapsack_greedy(
            self.weights, self.values, capacity
        )
        
        selected_items = []
        total_weight = 0
        for idx, fraction in selected_fractions:
            weight_used = self.weights[idx] * fraction
            value_used = self.values[idx] * fraction
            selected_items.append({
                'name': self.names[idx],
                'weight': self.weights[idx],
                'value': self.values[idx],
                'fraction': fraction,
                'weight_used': weight_used,
                'value_used': value_used
            })
            total_weight += weight_used
        
        return {
            'max_value': max_value,
            'total_weight': total_weight,
            'selected_items': selected_items
        }


# 测试示例
def test_knapsack_problems():
    print("=" * 50)
    print("背包问题测试示例")
    print("=" * 50)
    
    # 测试数据
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    names = ['A', 'B', 'C', 'D']
    capacity = 8
    
    print(f"物品信息：")
    for i in range(len(weights)):
        print(f"  {names[i]}: 重量={weights[i]}, 价值={values[i]}")
    print(f"背包容量: {capacity}")
    print()
    
    # 测试0-1背包问题
    print("1. 0-1背包问题")
    solver = KnapsackSolver(weights, values, names)
    result = solver.solve_01_knapsack(capacity)
    print(f"最大价值: {result['max_value']}")
    print(f"总重量: {result['total_weight']}")
    print("选择的物品:")
    for item in result['selected_items']:
        print(f"  {item['name']}: 重量={item['weight']}, 价值={item['value']}")
    print()
    
    # 测试分数背包问题
    print("2. 分数背包问题")
    result = solver.solve_fractional_knapsack(capacity)
    print(f"最大价值: {result['max_value']:.2f}")
    print(f"总重量: {result['total_weight']:.2f}")
    print("选择的物品:")
    for item in result['selected_items']:
        print(f"  {item['name']}: 重量={item['weight']}, 价值={item['value']}, "
              f"比例={item['fraction']:.2f}, 实际重量={item['weight_used']:.2f}")
    print()
    
    # 测试完全背包问题
    print("3. 完全背包问题")
    max_value = complete_knapsack(weights, values, capacity)
    print(f"最大价值: {max_value}")
    print()
    
    # 测试多重背包问题
    print("4. 多重背包问题")
    counts = [2, 1, 3, 2]  # 每个物品的数量
    max_value = multiple_knapsack(weights, values, counts, capacity)
    print(f"最大价值: {max_value}")
    print()


# 可视化示例
def visualize_knapsack(weights, values, capacity):
    """可视化背包问题的解决过程"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充dp表
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # 打印dp表
    print("动态规划表 (dp[i][w]):")
    print("    ", end="")
    for w in range(capacity + 1):
        print(f"{w:3d}", end=" ")
    print()
    print("-" * (4 * (capacity + 2)))
    
    for i in range(n + 1):
        print(f"{i:2d} |", end="")
        for w in range(capacity + 1):
            print(f"{dp[i][w]:3d}", end=" ")
        print()


if __name__ == "__main__":
    # 运行测试
    test_knapsack_problems()
    
    # 可视化示例
    print("=" * 50)
    print("动态规划表可视化示例")
    print("=" * 50)
    weights = [2, 3, 4]
    values = [3, 4, 5]
    capacity = 5
    visualize_knapsack(weights, values, capacity)
```

## 五、算法复杂度分析

1. **0-1背包问题**： 时间复杂度：O(n × capacity)，其中n是物品数量 空间复杂度：基础版本O(n × capacity)，优化版本O(capacity)
2. **完全背包问题**： 时间复杂度：O(n × capacity) 空间复杂度：O(capacity)
3. **分数背包问题**： 时间复杂度：O(n log n)，主要是排序的开销 空间复杂度：O(n)

## 六、应用场景

1. **资源分配**：在有限资源下最大化收益
2. **投资组合优化**：在预算限制下选择投资项目
3. **切割问题**：材料切割以最大化利用
4. **调度问题**：任务调度以最大化完成价值
5. **密码学**：公钥密码系统中的子集和问题

## 七、优化技巧

1. **空间优化**：使用一维数组代替二维数组
2. **二进制拆分**：用于多重背包问题
3. **分支限界法**：用于大规模问题
4. **近似算法**：当不需要精确解时