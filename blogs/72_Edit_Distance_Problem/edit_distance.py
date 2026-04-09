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
