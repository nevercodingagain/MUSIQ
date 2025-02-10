import os
import csv

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def create_log(self, log_name, columns):
        """创建新的日志文件"""
        log_path = os.path.join(self.log_dir, log_name)
        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        return log_path
    
    @staticmethod
    def add_record(log_path, record_dict):
        """添加记录到指定日志"""
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([record_dict.get(col, '') for col in record_dict])