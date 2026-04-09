## 1. 问题定义
给定两个字符串：  
- 源字符串 $A = \\text{"abcde"}$（长度为 $m = 5$）  
- 目标字符串 $B = \\text{"daedf"}$（长度为 $n = 5$）  

允许的操作（每次算一步）：  
1. **替换**：将 $B$ 中的一个字符改成 $A$ 中对应位置的字符（替换成本 $1$）  
2. **删除**：删掉 $B$ 中的一个字符（成本 $1$）  
3. **插入**：在 $B$ 中插入一个字符（成本 $1$）  

求从 $B$ 变成 $A$ 的最小操作步数。  

## 2. 动态规划定义
设 $dp[i][j]$ 表示从 $B$ 的前 $j$ 个字符（$B[0:j]$）变成 $A$ 的前 $i$ 个字符（$A[0:i]$）的最小编辑距离。  
这里 $i$ 是 $A$ 的长度索引（$0..m$），$j$ 是 $B$ 的长度索引（$0..n$）。  

- 当 $i = 0$ 时，$A$ 为空，需要删除 $B$ 的所有 $j$ 个字符，$dp[0][j] = j$  
- 当 $j = 0$ 时，$B$ 为空，需要插入 $A$ 的所有 $i$ 个字符，$dp[i][0] = i$  

转移方程（$i, j \\geq 1$）：  
1. 如果 $A[i-1] = B[j-1]$，则 $dp[i][j] = dp[i-1][j-1]$（无需操作）  
2. 如果 $A[i-1] \\neq B[j-1]$，则考虑三种操作的最小值：  
   - 插入：$dp[i][j-1] + 1$（在 $B$ 的前 $j-1$ 匹配 $A$ 的前 $i$ 后，在 $B$ 中插入 $A[i-1]$）  
   - 删除：$dp[i-1][j] + 1$（$B$ 的前 $j$ 匹配 $A$ 的前 $i-1$ 后，删除 $B$ 的一个字符）  
   - 替换：$dp[i-1][j-1] + 1$（将 $B[j-1]$ 替换成 $A[i-1]$）  

所以：  

$$
dp[i][j] = 
\\begin{cases}
dp[i-1][j-1] & \\text{if } A[i-1] = B[j-1] \\\\
\\min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 & \\text{otherwise}
\\end{cases}
$$

## 3. 示例代码

```python
def solve():
    # 读取字符串A，B
    A, B = input().strip().split()
    len_A = len(A)
    len_B = len(B)
    
    # dp矩阵是(len_A+1)行 x (len_B+1)列
    # 定义dp[i][j]表示A的前i个字符和B的前j个字符的编辑距离
    dp = [[0] * (len_B + 1) for _ in range(len_A + 1)]
    
    # 初始化
    for i in range(len_A + 1):
        dp[i][0] = i  # 从空字符串B变成A[0:i]需要i次插入
    for j in range(len_B + 1):
        dp[0][j] = j  # 从B[0:j]变成空字符串A需要j次删除
    
    # 动态规划填表
    for i in range(1, len_A + 1):
        for j in range(1, len_B + 1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 删除A[i-1]（或理解为在B中插入）
                    dp[i][j-1],      # 删除B[j-1]（或理解为在A中插入）
                    dp[i-1][j-1]     # 替换
                )
    
    # 输出结果
    print(dp[len_A][len_B])

if __name__ == '__main__':
    solve()
```

