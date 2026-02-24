import torch
import kvikio
import os
import time
import argparse
from typing import Tuple, List
import numpy as np

def setup_test_directory(base_dir='/RAID0/test_kvikio'):
    """设置测试目录"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def create_test_tensors(batch_size=8, seq_length=2048, hidden_size=4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建测试用的hidden_states和attention_mask张量"""
    # 创建bfloat16类型的hidden_states
    hidden_states = torch.randn(
        (batch_size, seq_length, hidden_size),
        dtype=torch.bfloat16,
        device='cuda:0'
    ).contiguous()
    
    # 创建bool类型的attention_mask
    attention_mask = torch.ones(
        (batch_size, seq_length),
        dtype=torch.bool,
        device='cuda:0'
    ).contiguous()
    
    # 添加一些随机模式以便验证
    for i in range(batch_size):
        for j in range(0, seq_length, 64):
            attention_mask[i, j:j+32] = False
    
    return hidden_states, attention_mask

def test_write_async(tensor, filepath, stream=None):
    """测试异步写入操作"""
    start_time = time.perf_counter()
    
    try:
        with kvikio.CuFile(filepath, "w") as f:
            if stream:
                future = f.raw_write_async(tensor, stream.cuda_stream)
            else:
                future = f.raw_write_async(tensor)
            
            bytes_written = future.check_bytes_done()
            expected_bytes = tensor.numel() * tensor.element_size()
            
            end_time = time.perf_counter()
            
            print(f"写入文件: {filepath}")
            print(f"张量形状: {tensor.shape}, 数据类型: {tensor.dtype}")
            print(f"预期字节数: {expected_bytes}, 实际写入字节数: {bytes_written}")
            print(f"写入耗时: {(end_time - start_time) * 1000:.2f} ms")
            print(f"写入速度: {bytes_written / (1024 * 1024) / (end_time - start_time):.2f} MB/s")
            
            if bytes_written != expected_bytes:
                print(f"警告: 写入字节数与预期不符!")
                
            return bytes_written == expected_bytes
    
    except Exception as e:
        print(f"写入错误: {e}")
        return False

def test_read_async(tensor_shape, tensor_dtype, filepath, stream=None):
    """测试异步读取操作"""
    start_time = time.perf_counter()
    
    try:
        # 创建GPU上的空张量接收数据
        buffer = torch.empty(tensor_shape, dtype=tensor_dtype, device='cuda:0')
        
        with kvikio.CuFile(filepath, "r") as f:
            if stream:
                future = f.raw_read_async(buffer, stream.cuda_stream)
            else:
                future = f.raw_read_async(buffer)
            
            bytes_read = future.check_bytes_done()
            expected_bytes = buffer.numel() * buffer.element_size()
            
            end_time = time.perf_counter()
            
            print(f"读取文件: {filepath}")
            print(f"张量形状: {buffer.shape}, 数据类型: {buffer.dtype}")
            print(f"预期字节数: {expected_bytes}, 实际读取字节数: {bytes_read}")
            print(f"读取耗时: {(end_time - start_time) * 1000:.2f} ms")
            print(f"读取速度: {bytes_read / (1024 * 1024) / (end_time - start_time):.2f} MB/s")
            
            if bytes_read != expected_bytes:
                print(f"警告: 读取字节数与预期不符!")
                
            return buffer, bytes_read == expected_bytes
    
    except Exception as e:
        print(f"读取错误: {e}")
        return None, False

def verify_data_integrity(original_tensor, read_tensor):
    """验证数据完整性"""
    if original_tensor.shape != read_tensor.shape:
        print(f"形状不匹配: 原始={original_tensor.shape}, 读取={read_tensor.shape}")
        return False
        
    if original_tensor.dtype != read_tensor.dtype:
        print(f"数据类型不匹配: 原始={original_tensor.dtype}, 读取={read_tensor.dtype}")
        return False
    
    is_equal = torch.all(original_tensor == read_tensor).item()
    if is_equal:
        print("数据完整性验证: 通过 ✓")
    else:
        # 计算差异
        if original_tensor.dtype == torch.bool:
            diff_count = (original_tensor != read_tensor).sum().item()
            total = original_tensor.numel()
            print(f"数据完整性验证: 失败 ✗ - {diff_count}/{total} 元素不匹配 ({diff_count/total*100:.2f}%)")
        else:
            # 对于浮点数，计算相对误差
            abs_diff = torch.abs(original_tensor - read_tensor)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()
            print(f"数据完整性验证: 失败 ✗ - 最大差异={max_diff}, 平均差异={mean_diff}")
    
    return is_equal

def test_concurrent_operations(hidden_states, attention_mask, test_dir):
    """测试并发读写操作"""
    print("\n===== 测试并发读写操作 =====")
    
    # 创建多个CUDA流
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # 准备文件路径
    hidden_path = os.path.join(test_dir, "concurrent_hidden_states")
    mask_path = os.path.join(test_dir, "concurrent_attention_mask")
    
    # 并发写入
    with torch.cuda.stream(stream1):
        write_success1 = test_write_async(hidden_states, hidden_path, stream1)
    
    with torch.cuda.stream(stream2):
        write_success2 = test_write_async(attention_mask, mask_path, stream2)
    
    # 同步流
    stream1.synchronize()
    stream2.synchronize()
    
    if not (write_success1 and write_success2):
        print("并发写入测试失败，跳过并发读取测试")
        return False
    
    # 并发读取
    with torch.cuda.stream(stream1):
        read_hidden, read_success1 = test_read_async(
            hidden_states.shape, hidden_states.dtype, hidden_path, stream1
        )
    
    with torch.cuda.stream(stream2):
        read_mask, read_success2 = test_read_async(
            attention_mask.shape, attention_mask.dtype, mask_path, stream2
        )
    
    # 同步流
    stream1.synchronize()
    stream2.synchronize()
    
    if not (read_success1 and read_success2):
        print("并发读取测试失败")
        return False
    
    # 验证数据
    integrity1 = verify_data_integrity(hidden_states, read_hidden)
    integrity2 = verify_data_integrity(attention_mask, read_mask)
    
    return integrity1 and integrity2

def run_sequential_tests(hidden_states, attention_mask, test_dir):
    """运行顺序读写测试"""
    print("\n===== 测试顺序读写操作 =====")
    
    # 准备文件路径
    hidden_path = os.path.join(test_dir, "hidden_states")
    mask_path = os.path.join(test_dir, "attention_mask")
    
    # 写入测试
    print("\n--- 写入测试 ---")
    write_success1 = test_write_async(hidden_states, hidden_path, torch.cuda.default_stream())
    write_success2 = test_write_async(attention_mask, mask_path, torch.cuda.default_stream())
    
    if not (write_success1 and write_success2):
        print("写入测试失败，跳过读取测试")
        return False
    
    # 读取测试
    print("\n--- 读取测试 ---")
    read_hidden, read_success1 = test_read_async(
        hidden_states.shape, hidden_states.dtype, hidden_path, torch.cuda.default_stream()
    )
    read_mask, read_success2 = test_read_async(
        attention_mask.shape, attention_mask.dtype, mask_path, torch.cuda.default_stream()
    )
    
    if not (read_success1 and read_success2):
        print("读取测试失败")
        return False
    
    # 验证数据完整性
    print("\n--- 数据完整性测试 ---")
    integrity1 = verify_data_integrity(hidden_states, read_hidden)
    integrity2 = verify_data_integrity(attention_mask, read_mask)
    
    return integrity1 and integrity2

def test_event_synchronization():
    """测试使用CUDA事件进行同步"""
    print("\n===== 测试CUDA事件同步 =====")
    
    # 创建事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 创建流
    stream = torch.cuda.Stream()
    
    # 记录开始事件
    start_event.record(stream=stream)
    
    # 在流上执行一些操作
    with torch.cuda.stream(stream):
        # 创建一个大张量并执行一些计算
        x = torch.randn(1000, 1000, device='cuda:0')
        for _ in range(100):
            x = x + x
    
    # 记录结束事件
    end_event.record(stream=stream)
    
    # 同步事件
    end_event.synchronize()
    
    # 计算时间
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"CUDA事件同步耗时: {elapsed_time:.2f} ms")
    
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试kvikio异步读写操作')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--seq_length', type=int, default=2048, help='序列长度')
    parser.add_argument('--hidden_size', type=int, default=4096, help='隐藏层大小')
    parser.add_argument('--test_dir', type=str, default='/RAID0/test_kvikio', help='测试目录')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 设置测试目录
    test_dir = setup_test_directory(args.test_dir)
    print(f"测试目录: {test_dir}")
    
    # 创建测试张量
    print("\n创建测试张量...")
    hidden_states, attention_mask = create_test_tensors(
        args.batch_size, args.seq_length, args.hidden_size
    )
    print(f"Hidden states: 形状={hidden_states.shape}, 类型={hidden_states.dtype}")
    print(f"Attention mask: 形状={attention_mask.shape}, 类型={attention_mask.dtype}")
    
    # 运行顺序测试
    sequential_success = run_sequential_tests(hidden_states, attention_mask, test_dir)
    print(f"顺序测试结果: {'成功' if sequential_success else '失败'}")
    
    # 运行并发测试
    concurrent_success = test_concurrent_operations(hidden_states, attention_mask, test_dir)
    print(f"并发测试结果: {'成功' if concurrent_success else '失败'}")
    
    # 测试事件同步
    event_success = test_event_synchronization()
    print(f"事件同步测试结果: {'成功' if event_success else '失败'}")
    
    # 总结
    print("\n===== 测试总结 =====")
    all_success = sequential_success and concurrent_success and event_success
    print(f"所有测试: {'全部通过 ✓' if all_success else '存在失败 ✗'}")

if __name__ == "__main__":
    main()
