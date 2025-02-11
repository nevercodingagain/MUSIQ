import os
import csv

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def create_log(self, log_name, columns):
        log_path = os.path.join(self.log_dir, log_name)
        # 检测文件是否存在并读取已有列
        file_exists = os.path.isfile(log_path)
        mode = 'a' if file_exists else 'w'
        
        with open(log_path, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists or (file_exists and os.stat(log_path).st_size == 0):
                writer.writerow(columns)
        return log_path
    
    def clean_logs(self, log_dir, resume_epoch):
        """清理训练和验证日志中指定epoch之后的记录"""
        log_files = ['train_log.csv', 'val_log.csv']
        
        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            if not os.path.exists(log_path):
                continue
                
            temp_path = log_path + ".tmp"
            with open(log_path, 'r') as fin, open(temp_path, 'w') as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                
                # 保留header
                header = next(reader)
                writer.writerow(header)
                
                # 过滤记录
                for row in reader:
                    if len(row) > 0 and row[0].isdigit():
                        current_epoch = int(row[0])
                        if current_epoch <= resume_epoch:
                            writer.writerow(row)
            
            # 原子替换文件
            os.replace(temp_path, log_path)
    
    @staticmethod
    def add_record(log_path, record_dict):
        """添加记录到指定日志"""
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([record_dict.get(col, '') for col in record_dict])