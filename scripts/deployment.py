import os

'''
如果在 git clone命令 运行过程中报错：
GnuTLS recv error (-110): The TLS connection was non-properly terminated
取消代理即可恢复正常，执行下面的命令：
git config --global --unset http.https://github.com.proxy
'''

def deploy_code():
    os.chdir('/var/www/html')
    os.system('git pull')

if __name__ == '__main__':
    deploy_code()