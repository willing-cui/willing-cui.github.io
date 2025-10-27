import os
import subprocess
import signal
import time
from typing import List, Dict

'''
如果在 git clone命令 运行过程中报错：
GnuTLS recv error (-110): The TLS connection was non-properly terminated
取消代理即可恢复正常，执行下面的命令：
git config --global --unset http.https://github.com.proxy

Ubuntu 需要安装 Python 虚拟环境来安装必要的软件包
sudo apt install python3-venv
# 创建一个新的虚拟环境，命名为app
python3 -m venv app
'''

class TaskManager:
    def __init__(self, base_path: str = '/var/www/html'):
        self.base_path = base_path
        self.scripts_path = os.path.join(base_path, 'scripts')
        self.venv_path = os.path.join(self.scripts_path, 'app/bin/activate')
        
        # 定义需要管理的任务列表
        # 可以轻松添加新任务，格式为 {'name': '任务名称', 'script': '脚本文件名', 'args': '参数'}
        self.tasks = [
            {'name': '热词爬取', 'script': 'hot_words.py', 'args': ''},
            {'name': '饰品金价查询', 'script': 'gold_price.py', 'args': ''},
            # 在这里添加新任务，例如：
            # {'name': '数据清洗', 'script': 'data_clean.py', 'args': '--mode=daily'},
            # {'name': 'API服务', 'script': 'api_server.py', 'args': '--port=8080'},
        ]
    
    def stop_all_tasks(self, graceful_timeout: int = 10):
        """停止所有正在运行的任务"""
        print("正在停止所有运行中的任务...")
        
        for task in self.tasks:
            script_name = task['script']
            self._stop_task(script_name, graceful_timeout)
    
    def _stop_task(self, script_name: str, graceful_timeout: int):
        """停止特定任务"""
        try:
            # 查找相关进程
            result = subprocess.run(
                ['pgrep', '-f', f'python3.*{script_name}'],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        pid = pid.strip()
                        print(f"正在停止任务 {script_name} (PID: {pid})...")
                        
                        # 先尝试优雅停止
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            continue
                
                # 等待优雅停止
                start_time = time.time()
                while time.time() - start_time < graceful_timeout:
                    # 检查进程是否还存在
                    check_result = subprocess.run(
                        ['pgrep', '-f', f'python3.*{script_name}'],
                        capture_output=True, text=True, check=False
                    )
                    
                    if check_result.returncode != 0:
                        print(f"任务 {script_name} 已优雅停止")
                        break
                    
                    time.sleep(1)
                else:
                    # 强制停止
                    print(f"任务 {script_name} 超时，强制停止...")
                    subprocess.run(['pkill', '-9', '-f', f'python3.*{script_name}'])
            
            else:
                print(f"任务 {script_name} 未在运行")
                
        except Exception as e:
            print(f"停止任务 {script_name} 时出错: {e}")
    
    def start_all_tasks(self):
        """启动所有任务"""
        print("正在启动所有任务...")
        
        for task in self.tasks:
            self._start_task(task)
    
    def _start_task(self, task: Dict):
        """启动特定任务"""
        script_path = os.path.join(self.scripts_path, task['script'])
        
        if not os.path.exists(script_path):
            print(f"警告: 脚本文件不存在: {script_path}")
            return
        
        # 构建启动命令
        activate_cmd = f'source {self.venv_path}'
        run_cmd = f'python3 {script_path} {task.get("args", "")}'.strip()
        full_cmd = f'{activate_cmd} && {run_cmd}'
        
        print(f"启动任务: {task['name']} ({task['script']})")
        
        try:
            # 使用 nohup 在后台运行，并重定向输出到日志文件
            log_file = os.path.join(self.scripts_path, f"logs/{task['script']}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # 在后台运行任务
            subprocess.Popen(
                f'{full_cmd} >> {log_file} 2>&1 &',
                shell=True,
                executable='/bin/bash',
                cwd=self.scripts_path
            )
            
            print(f"任务 {task['name']} 启动成功")
            time.sleep(2)  # 短暂延迟，避免同时启动所有任务造成资源竞争
            
        except Exception as e:
            print(f"启动任务 {task['name']} 失败: {e}")
    
    def check_task_status(self):
        """检查所有任务的运行状态"""
        print("检查任务状态...")
        
        for task in self.tasks:
            script_name = task['script']
            try:
                result = subprocess.run(
                    ['pgrep', '-f', f'python3.*{script_name}'],
                    capture_output=True, text=True, check=False
                )
                
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    print(f"✓ {task['name']} 正在运行 (PID: {', '.join(pids)})")
                else:
                    print(f"✗ {task['name']} 未运行")
                    
            except Exception as e:
                print(f"检查任务 {task['name']} 状态时出错: {e}")
    
    def deploy(self):
        """执行完整的部署流程"""
        print("开始部署流程...")
        
        # 切换到项目目录
        os.chdir(self.base_path)
        
        # 拉取最新代码
        print("拉取最新代码...")
        os.system('git pull')
        
        # 停止所有任务
        self.stop_all_tasks()
        
        # 等待确保所有进程已停止
        time.sleep(3)
        
        # 启动所有任务
        self.start_all_tasks()
        
        # 检查任务状态
        time.sleep(5)  # 给任务一些启动时间
        self.check_task_status()
        
        print("部署完成！")

def main():
    # 创建任务管理器实例
    task_manager = TaskManager()
    
    # 执行部署
    task_manager.deploy()

if __name__ == '__main__':
    main()