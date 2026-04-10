## 一、输入读取

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

## 二、lambda函数

`lambda`是 Python 中的一个关键字，用于创建**匿名函数**（即没有名称的函数）。它通常用于需要一个简单函数作为参数的场合，而不必用 `def`正式定义函数。

### 1. 基本语法

```python
lambda 参数1, 参数2, ...: 表达式
```

- `lambda`后面跟参数（可以有多个，用逗号分隔）
- 冒号 `:`后面是一个**表达式**（而不是代码块），这个表达式的结果就是函数的返回值
- 不需要写 `return`

### 2. 简单示例

```python
# 普通函数定义
def square(x):
    return x ** 2

# lambda 等价写法
square_lambda = lambda x: x ** 2

print(square(5))          # 25
print(square_lambda(5))   # 25
```

### 3. 常见使用场景

#### 3.1 与 `sorted()`、`sort()`一起使用

```python
students = [
    ('Alice', 88),
    ('Bob', 72),
    ('Charlie', 95)
]

# 按分数排序
students.sort(key=lambda x: x[1])
print(students)  # [('Bob', 72), ('Alice', 88), ('Charlie', 95)]
```

#### 3.2 与 `filter()`一起使用

```python
nums = [1, 2, 3, 4, 5, 6]
even_nums = list(filter(lambda x: x % 2 == 0, nums))
print(even_nums)  # [2, 4, 6]
```

#### 3.3 与 `map()`一起使用

```python
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x ** 2, nums))
print(squared)  # [1, 4, 9, 16]
```

#### 3.4 在字典排序中

```python
data = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 20}]
sorted_data = sorted(data, key=lambda x: x['age'])
print(sorted_data)  # 按 age 排序
```

### 4. 多参数示例

```python
add = lambda a, b: a + b
print(add(3, 7))  # 10

# 也可以有默认参数
multiply = lambda x, y=2: x * y
print(multiply(5))    # 10
print(multiply(5, 3)) # 15
```

### 5. 注意事项

1. **简洁性**：`lambda` 函数应该只用于简单的操作。如果逻辑复杂，应该用 `def`定义普通函数。
2. **可读性**：过度使用 `lambda` 可能降低代码可读性。
3. **只能有一个表达式**：`lambda` 函数体只能是单个表达式，不能包含多个语句或复杂控制流（如循环、条件语句块，但可以用三元表达式）。

### 6. 三元表达式在 lambda 中

```python
# 返回两个数中较大的数
max_num = lambda a, b: a if a > b else b
print(max_num(10, 20))  # 20
```

### 7. 与普通函数的区别

| 特性   | lambda 函数      | def 定义的函数   |
| ------ | ---------------- | ---------------- |
| 名称   | 匿名             | 有名称           |
| 函数体 | 只能是单个表达式 | 可以包含多个语句 |
| 复杂度 | 适合简单操作     | 适合复杂逻辑     |
| 可读性 | 可能较低         | 通常更高         |

### 8. 总结

`lambda`是一个方便的工具，尤其适合**函数式编程**场景和**需要短小回调函数**的场合。但它不是 `def`的完全替代品，应根据实际情况选择使用。
