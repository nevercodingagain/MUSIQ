import os
import time


class GPUGet:
    def __init__(self, min_gpu_number=1, time_interval=5, required_mem=1000, max_power=200):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval
        self.required_mem = required_mem  # 单位 MiB
        self.max_power = max_power        # 单位 W

    def get_gpu_info(self):
        """解析 nvidia-smi 输出，返回 GPU 状态字典"""
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = {}
        for i in range(len(gpu_status) // 4):
            index = i * 4
            # 解析 GPU 状态、功率、显存
            status = str(gpu_status[index].split('   ')[2].strip())
            power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            memory_used = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            memory_total = int(gpu_status[index + 1].split('/')[1].split('M')[0].strip())
            gpu_dict[i] = {
                "status": status,
                "power": power,
                "memory_used": memory_used,
                "memory_total": memory_total,
                "memory_available": memory_total - memory_used
            }
        return gpu_dict

    def get_available_gpus(self):
        """动态检测可用 GPU，返回GPU列表如：[0, 1, 2, 3]"""
        available_gpus = []
        while len(available_gpus) < self.min_gpu_number:
            gpu_dict = self.get_gpu_info()
            current_available = [
                gpu_id for gpu_id, info in gpu_dict.items()
                if info["memory_available"] >= self.required_mem and info["power"] <= self.max_power
            ]
            if len(current_available) >= self.min_gpu_number:
                available_gpus = current_available
                break
            else:
                print(f"等待可用 GPU... 当前可用: {current_available}")
                time.sleep(self.time_interval)
        return available_gpus