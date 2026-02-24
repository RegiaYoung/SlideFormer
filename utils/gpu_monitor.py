import threading
import time
import statistics
from pynvml import *

class GPUMonitor:
    def __init__(self, device_index=0, sample_interval=0.05):
        """
        一个用于在独立线程中监控GPU利用率的类。

        Args:
            device_index (int): 要监控的GPU设备索引。
            sample_interval (float): 采样间隔（秒），例如0.05代表每50毫秒采样一次。
        """
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.utilization_samples = []
        
        self._monitoring = False
        self._thread = None
        
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.device_index)
            print("pynvml initialized successfully.")
        except NVMLError as error:
            print(f"Failed to initialize NVML: {error}")
            raise

    def _monitor_loop(self):
        """监控线程的主循环。"""
        while self._monitoring:
            try:
                utilization = nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.utilization_samples.append(utilization)
            except NVMLError as error:
                print(f"Error getting GPU utilization: {error}")
                # 可以在这里选择停止循环
                break
            time.sleep(self.sample_interval)

    def start(self):
        """开始监控。"""
        if self._thread is not None and self._thread.is_alive():
            print("Monitor is already running.")
            return

        print(f"Starting GPU monitor for device {self.device_index}...")
        self.utilization_samples = []  # 清空上一次的记录
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()

    def stop(self):
        """停止监控并等待线程结束。"""
        if not self._monitoring:
            print("Monitor is not running.")
            return
        
        print("Stopping GPU monitor...")
        self._monitoring = False
        self._thread.join() # 等待线程完全退出
        print("GPU monitor stopped.")

    def get_average_utilization(self):
        """计算并返回平均利用率。"""
        if not self.utilization_samples:
            return 0.0
        return statistics.mean(self.utilization_samples)

    def __del__(self):
        """确保在对象销毁时关闭NVML。"""
        try:
            nvmlShutdown()
        except:
            pass