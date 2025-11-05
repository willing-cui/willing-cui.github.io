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

class GoldPriceCollector:
    """金价数据收集器基类"""
    
    def __init__(self):
        self.HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.chowtaifook.com/zh-hk/eshop/realtime-gold-price.html'
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

class ChowTaiFookGoldPrice(GoldPriceCollector):
    """周大福金价数据采集"""
    
    def fetch(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """获取周大福金价数据"""
        try:
            url = "https://www.chowtaifook.com/zh-hk/eshop/realtime-gold-price.html"
            response = self.make_request(url)
            if not response:
                return self.standardize_result('周大福', error='请求失败')
            
            # 解析HTML内容
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找包含金价数据的隐藏输入字段
            gold_price_data_input = soup.find('input', {'class': 'gold-price-data'})
            if not gold_price_data_input:
                return self.standardize_result('周大福', error='未找到金价数据')
            
            # 提取JSON数据
            gold_price_json = gold_price_data_input.get('value', '{}')
            gold_price_data = json.loads(gold_price_json)
            
            # 提取更新时间
            updated_time = gold_price_data.get('Updated_Time', '')
            
            # 构建金价数据列表
            gold_items = []
            
            # 添加各种金价类型
            gold_types = [
                {
                    'name': '999.9饰金卖出价',
                    'price_per_tael': gold_price_data.get('Gold_Sell', ''),
                    'price_per_gram': gold_price_data.get('Gold_Sell_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '999.9饰金买入价',
                    'price_per_tael': gold_price_data.get('Gold_Buy', ''),
                    'price_per_gram': gold_price_data.get('Gold_Buy_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '饰金换金价',
                    'price_per_tael': gold_price_data.get('Redemption_Price', ''),
                    'price_per_gram': gold_price_data.get('Redemption_Price_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '饰金换珠宝价',
                    'price_per_tael': gold_price_data.get('Jewellery_Redemption_Price', ''),
                    'price_per_gram': gold_price_data.get('Jewellery_Redemption_Price_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '金粒卖出价',
                    'price_per_tael': gold_price_data.get('Gold_Pellet_Sell', ''),
                    'price_per_gram': gold_price_data.get('Gold_Pellet_Sell_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '金粒买入价',
                    'price_per_tael': gold_price_data.get('Gold_Pellet_Buy', ''),
                    'price_per_gram': gold_price_data.get('Gold_Pellet_Buy_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '金粒换货价',
                    'price_per_tael': gold_price_data.get('Gold_Pellet_Redemption_Price', ''),
                    'price_per_gram': gold_price_data.get('Gold_Pellet_Redemption_Price_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '足铂铂金',
                    'price_per_tael': gold_price_data.get('Platinum', ''),
                    'price_per_gram': gold_price_data.get('Platinum_g', ''),
                    'currency': 'HKD'
                },
                {
                    'name': '足铂铂金换货价',
                    'price_per_tael': gold_price_data.get('Platinum_Redemption_Price', ''),
                    'price_per_gram': gold_price_data.get('Platinum_Redemption_Price_g', ''),
                    'currency': 'HKD'
                }
            ]
            
            for gold_type in gold_types:
                if gold_type['price_per_tael'] and gold_type['price_per_gram']:
                    gold_items.append({
                        'type': gold_type['name'],
                        'price_per_tael': f"{gold_type['price_per_tael']} {gold_type['currency']}",
                        'price_per_gram': f"{gold_type['price_per_gram']} {gold_type['currency']}",
                        'updated_time': updated_time
                    })
            
            return self.standardize_result('周大福', data=gold_items)
        
        except Exception as e:
            print(f"周大福金价解析失败: {e}")
            return self.standardize_result('周大福', error=str(e))

class DataManager:
    """数据管理类"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or GoldPriceCollector.get_script_dir()
        self.data_dir = os.path.join(self.base_dir, 'gold_prices')
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
                    for item in result['data']:
                        f.write(f"{item['type']}:\n")
                        f.write(f"  每两: {item['price_per_tael']}\n")
                        f.write(f"  每克: {item['price_per_gram']}\n")
                        f.write(f"  更新时间: {item['updated_time']}\n")
                        f.write("\n")
                else:
                    f.write(f"获取失败: {result.get('error', '未知错误')}\n")
                f.write("\n")
        except Exception as e:
            print(f"保存文本文件失败: {filename} - {e}")
    
    def _save_to_json(self, results: List[Dict], date: str, timestamp: str) -> None:
        """保存为JSON文件"""
        filename = os.path.join(self.data_dir, f"gold_prices_{date}.json")
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
    
    def clean_old_files(self, days: int = 365) -> int:
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

class GoldPriceMonitor:
    """金价监控主程序"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.collectors = {
            '周大福': ChowTaiFookGoldPrice()
        }
        self.is_server = ChowTaiFookGoldPrice().is_server_environment()
        
    def log(self, message: str) -> None:
        """日志记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        if self.is_server:
            log_file = os.path.join(self.data_manager.data_dir, 'gold_price_monitor.log')
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {message}\n")
            except:
                pass
    
    def collect_gold_prices(self) -> List[Dict]:
        """收集金价数据"""
        self.log("开始收集金价数据...")
        results = []
        
        for name, collector in self.collectors.items():
            try:
                start_time = time.time()
                result = collector.fetch()
                elapsed = time.time() - start_time
                
                self.log(f"{name}金价数据收集完成，耗时{elapsed:.2f}秒")
                results.append(result)
                
                if self.is_server:
                    time.sleep(2)
                    
            except Exception as e:
                self.log(f"{name}金价数据收集异常: {e}")
                results.append(collector.standardize_result(name, error=str(e)))
        
        return results
    
    def run_job(self) -> None:
        """定时任务"""
        try:
            results = self.collect_gold_prices()
            self.data_manager.save_results(results)
            
            success_count = sum(1 for r in results if r['success'])
            self.log(f"金价数据收集完成，成功{success_count}/{len(results)}个平台")
            
            # 每天凌晨清理旧数据
            if datetime.now().hour == 3:
                self.data_manager.clean_old_files()
                
        except Exception as e:
            self.log(f"定时任务执行失败: {e}")
    
    def start(self, immediate: bool = True) -> None:
        """启动监控"""
        self.log("金价监控服务启动")
        self.log(f"运行环境: {'服务器' if self.is_server else '本地'}")
        
        if immediate:
            self.run_job()
        
        # 设置定时任务：每小时执行一次
        interval = 1 if self.is_server else 0.01
        schedule.every(interval).hours.do(self.run_job)
        
        # 每天凌晨3点清理旧数据
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
        monitor = GoldPriceMonitor()
        monitor.start()
    except KeyboardInterrupt:
        print("\n程序被用户终止")
    except Exception as e:
        print(f"程序异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()