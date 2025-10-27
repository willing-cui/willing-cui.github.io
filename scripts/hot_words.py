import os
import sys
import time
import json
import glob
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

import requests
from bs4 import BeautifulSoup
from lxml import etree

"""
Required packages:
pip install requests beautifulsoup4 schedule lxml
"""

class HotWordCollector:
    """网络热词收集器基类"""
    
    def __init__(self):
        self.HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://weibo.com'
        }
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
    @staticmethod
    def get_script_dir() -> str:
        """获取脚本所在目录"""
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    
    def is_server_environment(self) -> bool:
        """检测是否为服务器环境"""
        return sys.platform == 'linux'
    
    def make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """封装请求方法"""
        try:
            max_retries = 3 if self.is_server_environment() else 1
            for attempt in range(max_retries):
                try:
                    response = self.session.request(method, url, timeout=15, **kwargs)
                    response.raise_for_status()
                    return response
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"请求失败: {url} - {str(e)}")
        return None
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        return text.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    
    def standardize_result(self, source: str, data: List[Dict] = None, error: str = None) -> Dict:
        """标准化返回格式"""
        return {
            'platform': source,
            'timestamp': datetime.now().isoformat(),
            'success': error is None,
            'data': data or [],
            'error': error,
            'count': len(data) if data else 0
        }

class BaiduHot(HotWordCollector):
    """百度热搜采集"""
    
    def fetch(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """获取百度热搜"""
        try:
            url = "https://top.baidu.com/board?tab=realtime"
            response = self.make_request(url)
            if not response:
                return self.standardize_result('百度', error='请求失败')
            
            soup = BeautifulSoup(response.text, 'html.parser')
            hot_items = []
            
            selectors = [
                'div.c-single-text-ellipsis',
                '.title-text',
                '.list-title'
            ]
            
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    for idx, item in enumerate(items[:20], 1):
                        word = self.clean_text(item.get_text())
                        if word and len(word) > 1:
                            hot_items.append({
                                'rank': str(idx),
                                'keyword': word,
                                'hot_value': '',
                                'tag': '',
                                'link': f"https://www.baidu.com/s?wd={word}"
                            })
                    break
            
            return self.standardize_result('百度', data=hot_items)
        
        except Exception as e:
            print(f"百度热搜解析失败: {e}")
            return self.standardize_result('百度', error=str(e))

class WeiboHot(HotWordCollector):
    """微博热搜采集"""
    
    def __init__(self):
        super().__init__()
    
    def fetch(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """获取微博热搜"""
        try:
            api_result = self.fetch_via_api()
            if api_result:
                return self.standardize_result('微博', data=api_result)
            return self.standardize_result('微博', error='API获取失败')
        except Exception as e:
            return self.standardize_result('微博', error=str(e))
    
    def fetch_via_api(self) -> Optional[List[Dict[str, str]]]:
        """通过API获取微博热搜"""
        api_endpoints = [
            "https://weibo.com/ajax/statuses/hot_band",
            "https://api.weibo.com/2/search/suggestions/hot_band.json",
        ]
        
        for api_url in api_endpoints:
            try:
                response = self.make_request(api_url)
                if response and response.status_code == 200:
                    data = response.json()
                    return self._parse_api_data(data)
            except Exception:
                continue
                
        return None
    
    def _parse_api_data(self, data: dict) -> List[Dict[str, str]]:
        """解析API返回数据"""
        hot_items = []
        
        if 'data' in data and 'band_list' in data['data']:
            for idx, item in enumerate(data['data']['band_list'][:50]):
                hot_items.append({
                    'rank': str(idx + 1),
                    'keyword': item.get('note', ''),
                    'hot_value': str(item.get('num', 0)),
                    'tag': item.get('category', ''),
                    'link': f"https://s.weibo.com/weibo?q=%23{item.get('word', '')}%23"
                })
        elif 'data' in data and isinstance(data['data'], list):
            for idx, item in enumerate(data['data'][:50]):
                if isinstance(item, dict):
                    hot_items.append({
                        'rank': str(idx + 1),
                        'keyword': item.get('title', item.get('keyword', '')),
                        'hot_value': str(item.get('hot_value', item.get('count', 0))),
                        'tag': item.get('tag', ''),
                        'link': item.get('url', '')
                    })
                    
        return hot_items

class ZhihuHot(HotWordCollector):
    """知乎热榜采集"""
    
    def fetch(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """获取知乎热榜"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
                'Host': 'api.zhihu.com',
            }
            params = {
                'limit': '50',
                'reverse_order': '0',
            }
            
            response = self.make_request(
                'https://zhihu.com/topstory/hot-list',
                headers=headers,
                params=params
            )
            
            if not response or response.status_code != 200:
                return self.standardize_result('知乎', error='请求失败')
            
            data = response.json()
            hot_items = []
            
            for idx, item in enumerate(data.get('data', [])[:20], 1):
                try:
                    target = item.get('target', {})
                    detail_text = item.get('detail_text', '0 万热度')
                    
                    hot_items.append({
                        'rank': str(idx),
                        'keyword': self.clean_text(target.get('title', '')),
                        'hot_value': detail_text.split()[0] + '万',
                        'tag': '',
                        'link': target.get('url', '').replace('api', 'www').replace('questions', 'question')
                    })
                except Exception as e:
                    continue
            
            return self.standardize_result('知乎', data=hot_items)
            
        except Exception as e:
            print(f"知乎热榜获取失败: {e}")
            return self.standardize_result('知乎', error=str(e))

class DataManager:
    """数据管理类"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or HotWordCollector.get_script_dir()
        self.data_dir = os.path.join(self.base_dir, 'hot_words')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_results(self, results: List[Dict], file_format: str = 'both') -> None:
        """保存结果到文件"""
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in results:
            if file_format in ('txt', 'both'):
                self._save_to_txt(result, today, now)
                
        if file_format in ('json', 'both'):
            self._save_to_json(results, today, now)
    
    def _save_to_txt(self, result: Dict, date: str, timestamp: str) -> None:
        """保存为文本文件"""
        filename = os.path.join(self.data_dir, f"{result['platform']}_{date}.txt")
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"=== {timestamp} ===\n")
                if result['success'] and result['data']:
                    for item in result['data'][:20]:
                        hot_info = f" ({item['hot_value']})" if item['hot_value'] else ""
                        f.write(f"{item['rank']}. {item['keyword']}{hot_info}\n")
                else:
                    f.write(f"获取失败: {result.get('error', '未知错误')}\n")
                f.write("\n")
        except Exception as e:
            print(f"保存文本文件失败: {filename} - {e}")
    
    def _save_to_json(self, results: List[Dict], date: str, timestamp: str) -> None:
        """保存为JSON文件"""
        filename = os.path.join(self.data_dir, f"all_{date}.json")
        try:
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_data.append(json.loads(line))
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            filtered_data = [
                item for item in existing_data 
                if datetime.strptime(item['time'], "%Y-%m-%d %H:%M:%S") > cutoff_time
            ]
            
            filtered_data.append({
                'time': timestamp,
                'results': results
            })
            
            with open(filename, 'w', encoding='utf-8') as f:
                for item in filtered_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
                    
        except Exception as e:
            print(f"保存JSON文件失败: {filename} - {e}")
    
    def clean_old_files(self, days: int = 7) -> int:
        """清理旧数据文件"""
        deleted_files = 0
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in glob.glob(os.path.join(self.data_dir, '*')):
            try:
                filename = os.path.basename(file_path)
                if '_' in filename and '.' in filename:
                    date_part = filename.split('_')[-1].split('.')[0]
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        deleted_files += 1
            except (ValueError, IndexError, Exception):
                continue
                
        print(f"清理完成，共删除 {deleted_files} 个过期文件")
        return deleted_files

class HotWordMonitor:
    """热词监控主程序"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.collectors = {
            '百度': BaiduHot(),
            '微博': WeiboHot(),
            '知乎': ZhihuHot()
        }
        self.is_server = BaiduHot().is_server_environment()
        
    def log(self, message: str) -> None:
        """日志记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        if self.is_server:
            log_file = os.path.join(self.data_manager.data_dir, 'monitor.log')
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {message}\n")
            except:
                pass
    
    def collect_hot_words(self) -> List[Dict]:
        """收集所有平台热词"""
        self.log("开始收集热词...")
        results = []
        
        for name, collector in self.collectors.items():
            try:
                start_time = time.time()
                result = collector.fetch()
                elapsed = time.time() - start_time
                
                self.log(f"{name}热词收集完成，耗时{elapsed:.2f}秒")
                results.append(result)
                
                if self.is_server:
                    time.sleep(2)
                    
            except Exception as e:
                self.log(f"{name}热词收集异常: {e}")
                results.append(collector.standardize_result(name, error=str(e)))
        
        return results
    
    def run_job(self) -> None:
        """定时任务"""
        try:
            results = self.collect_hot_words()
            self.data_manager.save_results(results)
            
            success_count = sum(1 for r in results if r['success'])
            self.log(f"热词收集完成，成功{success_count}/{len(results)}个平台")
            
            if datetime.now().hour == 3:
                self.data_manager.clean_old_files()
                
        except Exception as e:
            self.log(f"定时任务执行失败: {e}")
    
    def start(self, immediate: bool = True) -> None:
        """启动监控"""
        self.log("热词监控服务启动")
        self.log(f"运行环境: {'服务器' if self.is_server else '本地'}")
        
        if immediate:
            self.run_job()
        
        interval = 1 if self.is_server else 0.01
        schedule.every(interval).hours.do(self.run_job)
        
        schedule.every().day.at("03:00").do(self.data_manager.clean_old_files)
        
        self.log(f"数据保存目录: {self.data_manager.data_dir}")
        self.log(f"执行频率: 每{interval}小时一次")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                self.log("服务被用户中断")
                break
            except Exception as e:
                self.log(f"主循环异常: {e}")
                time.sleep(300)

def main():
    """主函数"""
    try:
        monitor = HotWordMonitor()
        monitor.start()
    except KeyboardInterrupt:
        print("\n程序被用户终止")
    except Exception as e:
        print(f"程序异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()