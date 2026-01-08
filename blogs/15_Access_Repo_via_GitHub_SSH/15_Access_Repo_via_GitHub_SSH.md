> 服务器代码拉取报错：error: RPC failed; curl 28 Failed to connect to github.com port 443 after 134825 ms: Couldn't connect to server
> fatal: expected flush after ref listing

这个错误通常是由于网络连接问题导致的（在防火墙设置正确时，也可能与云服务商的网络策略有关）。可以从 https 连接更换为 SSH 连接。

## 1. **检查服务器网络连接**

```
# 测试是否能ping通GitHub
ping github.com

# 测试443端口连接
nc -zv github.com 443

# 如果服务器无法访问GitHub，可能是网络策略问题
# 检查服务器安全组/防火墙是否放行443端口
```

## 2. **使用SSH方式替代HTTPS（推荐）**

**使用SSH协议通常能显著改善连接稳定性**，建议优先尝试这个方法。

### 第一步：在服务器生成SSH密钥

```
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 一路回车，使用默认位置
```

### 第二步：添加公钥到GitHub

```
cat ~/.ssh/id_rsa.pub
# 复制输出内容
```

- 登录GitHub → Settings（点击头像） → SSH and GPG keys → New SSH key
- 粘贴公钥内容

### 第三步：修改git远程地址

cd 到 git 仓库目录下

```
# 查看当前远程地址
git remote -v

# 如果是HTTPS地址，修改为SSH
git remote set-url origin git@github.com:你的用户名/仓库名.git

# 或首次克隆时使用SSH
git clone git@github.com:你的用户名/仓库名.git
```

## 3. **增加Git的超时时间（可选）**

```
# 临时设置
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git config --global core.compression 0

# 永久设置（编辑~/.gitconfig文件）
[core]
    compression = 0
[http]
    postBuffer = 524288000
    lowSpeedLimit = 0
    lowSpeedTime = 999999
```

