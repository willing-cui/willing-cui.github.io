# Server Setup

## Initial Setup

**安装Web服务器软件**

以Ubuntu为例，安装Nginx的命令是：

`sudo apt update && sudo apt install nginx`

**上传静态网页文件**

将HTML、CSS、JavaScript和图片等静态文件，通过SFTP或SCP工具上传到服务器的Web根目录。

- 通常情况下，这个目录是 `/var/www/html`。 

也可以配置项目**使用 Github 自动部署**：

1. 在项目根目录创建 `.github/workflows/deploy.yml`文件。

2. 配置 GitHub Actions 工作流，示例：

   ```yaml
   name: Deploy to Server
   
   on:
     push:
       branches: [ main ]
   
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Copy files to server
           uses: appleboy/scp-action@master
           with:
             host: ${{ secrets.SERVER_HOST }}
             username: ${{ secrets.SERVER_USER }}
             password: ${{ secrets.SERVER_PASSWORD }}
             source: "."
             target: "/var/www/html"
   ```

3. 在 GitHub 仓库的 `Settings > Secrets`中添加服务器凭据（`SERVER_HOST`, `SERVER_USER`, `SERVER_PASSWORD`）。

关于 Github 配置的其他参考：https://zhuanlan.zhihu.com/p/433426848

**配置和启动Nginx**

上传完成后，可以通过命令 `sudo systemctl start nginx` 来启动Nginx服务。 注意打开80端口。

