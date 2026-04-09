## 1. 输入读取

当命令行输入格式为

```text
100 200 300
```

可以通过下面的代码读取

```python
A, B, C = map(int, input().strip().split())
```

如果连续多行输入，并以回车换行

```text
100 200 300
400 500 600 700
```

可以通过连续调用上面的代码依此读取各行

```python
A, B, C = map(int, input().strip().split())
D, E, F, G = map(int, input().strip().split())
```

