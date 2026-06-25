# Copyright 2025-2026 The SlideFormer Authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import hashlib
import importlib
import platform
import subprocess
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load, CUDA_HOME

class CPUAdamLoader:
    """CPU Adam优化器加载器，仅为x86_64架构提供支持"""
    
    def __init__(self):
        # 验证CPU架构
        arch = platform.machine()
        if arch != "x86_64":
            raise RuntimeError(f"CPU Adam仅支持x86_64架构，但当前系统为{arch}")
    
    def get_extension_path(self):
        """获取扩展编译缓存路径"""
        # 获取torch版本
        torch_version = f"{torch.__version__.split('.')[0]}.{torch.__version__.split('.')[1]}"
        
        # 使用项目路径作为hash生成缓存目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        hash_suffix = hashlib.sha256(str(project_root).encode()).hexdigest()[:8]
        
        # 创建缓存目录
        home_dir = os.path.expanduser("~")
        cache_dir = f".cache/oom_optimizer/torch_extensions/torch{torch_version}_cpu-{hash_suffix}"
        full_path = os.path.join(home_dir, cache_dir)
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def get_sources(self):
        """获取C++源文件列表"""
        current_dir = Path(__file__).parent
        return [
            str(current_dir / "csrc" / "cpu_adam.cpp"),
            str(current_dir / "csrc" / "cpu_adam_impl.cpp")
        ]
    
    def get_include_dirs(self):
        """获取包含目录列表"""
        current_dir = Path(__file__).parent
        return [str(current_dir / "csrc")]
    
    def get_compile_args(self):
        """获取编译选项"""
        version_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]
        arch_flags = []
        
        # 检测CPU特性并添加相应的编译标志
        try:
            if self._has_avx512():
                arch_flags.append("-D__AVX512__")
            elif self._has_avx2():
                arch_flags.append("-D__AVX256__")
            else:
                arch_flags.append("-D__SCALAR__")
        except Exception:
            # 如果检测失败，默认使用标量实现
            arch_flags.append("-D__SCALAR__")
        
        compile_args = [
            "-O3",
            "-std=c++14",
            "-std=c++17",
            "-g",
            "-Wno-reorder",
            "-fopenmp",
            "-march=native"
        ]
        
        return version_macros + arch_flags + compile_args
    
    def _has_avx512(self):
        """检测是否支持AVX512指令集"""
        try:
            result = subprocess.run("lscpu | grep -q avx512", shell=True)
            return result.returncode == 0
        except:
            return False
    
    def _has_avx2(self):
        """检测是否支持AVX2指令集"""
        try:
            result = subprocess.run("lscpu | grep -q avx2", shell=True)
            return result.returncode == 0
        except:
            return False
    
    def load(self):
        """加载CPU Adam扩展"""
        # 首先尝试导入已安装的扩展
        try:
            return importlib.import_module("oom_optimizer._C.cpu_adam")
        except (ImportError, ModuleNotFoundError):
            pass
        
        # 如果没有找到已安装的扩展，则JIT编译
        build_dir = self.get_extension_path()
        
        # 检查是否已经编译过
        jit_module_path = os.path.join(build_dir, "cpu_adam.so")
        compiled_before = os.path.exists(jit_module_path)
        print(f"{'Loading' if compiled_before else 'Compiling'} CPU Adam optimizer...")
        
        start_time = time.time()
        op_module = load(
            name="cpu_adam",
            sources=self.get_sources(),
            extra_include_paths=self.get_include_dirs(),
            extra_cflags=self.get_compile_args(),
            build_directory=build_dir,
        )
        
        duration = time.time() - start_time
        print(f"CPU Adam {'Load' if compiled_before else 'Compile'} finished, cost {duration:.2f} seconds")
        
        return op_module

class FusedAdamLoader:
    """CUDA上的Fused Adam优化器加载器"""
    
    def __init__(self):
        # 验证CUDA或ROCm是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("FusedAdam需要CUDA或ROCm支持")
        if not self.is_rocm_pytorch() and CUDA_HOME is None:
            raise RuntimeError("未找到CUDA_HOME，请确保CUDA工具包已正确安装")
        
        # 初始化BF16支持标志
        self.enable_bf16 = False
        
    def get_extension_path(self):
        """获取扩展编译缓存路径"""
        # 获取torch版本
        torch_version = f"{torch.__version__.split('.')[0]}.{torch.__version__.split('.')[1]}"
        
        # 根据CUDA/ROCm环境确定后缀
        if self.is_rocm_pytorch():
            backend_version = f"hip{torch.version.hip.replace('.', '')}" if torch.version.hip else "hip"
        else:
            backend_version = f"cuda{torch.version.cuda.replace('.', '')}"
        
        # 使用项目路径作为hash生成缓存目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        hash_suffix = hashlib.sha256(str(project_root).encode()).hexdigest()[:8]
        
        # 创建缓存目录
        home_dir = os.path.expanduser("~")
        cache_dir = f".cache/oom_optimizer/torch_extensions/torch{torch_version}_{backend_version}-{hash_suffix}"
        full_path = os.path.join(home_dir, cache_dir)
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def get_sources(self):
        """获取源文件列表"""
        current_dir = Path(__file__).parent
        return [
            str(current_dir / "csrc" / "fused_adam_frontend.cpp"),
            str(current_dir / "csrc" / "multi_tensor_adam.cu")
        ]
    
    def get_include_dirs(self):
        """获取包含目录列表"""
        current_dir = Path(__file__).parent
        return [
            str(current_dir / "csrc")
        ]
    
    def get_version_dependent_macros(self):
        """获取与PyTorch版本相关的宏定义"""
        version_ge_1_1 = []
        if (int(torch.__version__.split('.')[0]) > 1) or (int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        
        version_ge_1_3 = []
        if (int(torch.__version__.split('.')[0]) > 1) or (int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        
        version_ge_1_5 = []
        if (int(torch.__version__.split('.')[0]) > 1) or (int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5
    
    def get_cxx_args(self):
        """获取C++编译选项"""
        if sys.platform == "win32":
            compile_args = ["-O2"]
        else:
            compile_args = ["-O3", "-std=c++17", "-g", "-Wno-reorder"]
        
        compile_args += self.get_version_dependent_macros()
        
        # 为ROCm添加特定的编译参数
        if self.is_rocm_pytorch():
            compile_args.append("-D__HIP_PLATFORM_AMD__=1")
            rocm_wavefront_size = self.get_rocm_wavefront_size()
            compile_args.append(f'-DROCM_WAVEFRONT_SIZE={rocm_wavefront_size}')
        
        # 添加BF16支持（如果可用）
        if self.enable_bf16:
            compile_args.append("-DBF16_AVAILABLE")
        
        return compile_args
    
    def installed_cuda_version(self):
        """获取已安装的CUDA版本"""
        if self.is_rocm_pytorch():
            return None, None
        
        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME does not exist, unable to compile CUDA op(s)")
        
        # 确保torch和nvcc编译器之间没有CUDA版本不匹配
        output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
        output_split = output.split()
        release_idx = output_split.index("release")
        release = output_split[release_idx + 1].replace(',', '').split(".")
        # 忽略补丁版本，只查看主要和次要版本
        cuda_major, cuda_minor = release[:2]
        return int(cuda_major), int(cuda_minor)
    
    def compute_capability_args(self):
        """返回NVCC计算能力编译标志"""
        # 检查环境变量中的TORCH_CUDA_ARCH_LIST
        ccs = []
        cross_compile_archs_env = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
        
        # 在JIT模式下，针对底层架构编译
        if cross_compile_archs_env is None:
            # 编译用于基础架构（我们在运行时知道）
            for i in range(torch.cuda.device_count()):
                CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability(i)
                cc = f"{CC_MAJOR}.{CC_MINOR}"
                if cc not in ccs:
                    ccs.append(cc)
            ccs = sorted(ccs)
            if ccs:  # 确保列表不为空
                ccs[-1] += '+PTX'
        else:
            # 使用环境变量中指定的计算能力
            ccs = cross_compile_archs_env.replace(' ', ';').split(';')
        
        # 如果没有获取到任何计算能力，使用默认值
        if not ccs:
            default_archs = ["6.0", "6.1", "7.0", "7.5", "8.0", "8.6"]
            # 根据CUDA版本添加更高的计算能力
            cuda_major, cuda_minor = self.installed_cuda_version()
            if cuda_major == 11:
                if cuda_minor >= 0:
                    default_archs.append("8.0")
                if cuda_minor >= 1:
                    default_archs.append("8.6")
                if cuda_minor >= 8:
                    default_archs.append("9.0")
            elif cuda_major == 12:
                default_archs.extend(["8.0", "8.6", "9.0"])
                if cuda_minor >= 8:
                    default_archs.extend(["10.0", "12.0"])
            ccs = default_archs
        
        # 过滤并格式化计算能力
        args = []
        for cc in ccs:
            parts = cc.split('.')
            if len(parts) < 2:
                continue
            
            num = parts[0] + parts[1].split('+')[0]
            args.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if '+PTX' in parts[1]:
                args.append(f'-gencode=arch=compute_{num},code=compute_{num}')
            
            # 根据计算能力确定BF16支持
            if int(parts[0]) > 7:
                self.enable_bf16 = True
        
        return args
    
    def get_nvcc_args(self):
        """获取NVCC编译选项"""
        if self.is_rocm_pytorch():
            nvcc_flags = ["-O3"]
            rocm_major, rocm_minor = self.installed_rocm_version()
            nvcc_flags += [
                "-std=c++17", 
                "-U__HIP_NO_HALF_OPERATORS__", 
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF2_OPERATORS__",
                f"-DROCM_VERSION_MAJOR={rocm_major}",
                f"-DROCM_VERSION_MINOR={rocm_minor}",
                f'-DROCM_WAVEFRONT_SIZE={self.get_rocm_wavefront_size()}'
            ]
        else:
            # CUDA特定的编译选项
            nvcc_flags = ["-O3"]
            
            # 获取CUDA版本并选择适当的C++标准
            cuda_major, cuda_minor = self.installed_cuda_version()
            if cuda_major > 10:
                if cuda_major == 12 and cuda_minor >= 5:
                    std_lib = '-std=c++20'
                else:
                    std_lib = '-std=c++17'
            else:
                std_lib = '-std=c++14'
            
            # 尝试获取NVCC线程数
            try:
                nvcc_threads = int(os.getenv("DS_NVCC_THREADS", ""))
                if nvcc_threads <= 0:
                    raise ValueError("")
            except ValueError:
                nvcc_threads = min(os.cpu_count(), 8)
            
            nvcc_flags += [
                '-allow-unsupported-compiler' if sys.platform == "win32" else '',
                '--use_fast_math',
                std_lib,
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                f'--threads={nvcc_threads}'
            ]
            
            # 添加调试信息（如果需要）
            if os.environ.get('DS_DEBUG_CUDA_BUILD', '0') == '1':
                nvcc_flags.append('--ptxas-options=-v')
            
            # 添加计算能力参数
            nvcc_flags += self.compute_capability_args()
            
            # 添加版本相关宏
            nvcc_flags += self.get_version_dependent_macros()
            
            # 添加BF16支持（如果可用）
            if self.enable_bf16:
                nvcc_flags.append("-DBF16_AVAILABLE")
                nvcc_flags.append("-U__CUDA_NO_BFLOAT16_OPERATORS__")
                nvcc_flags.append("-U__CUDA_NO_BFLOAT162_OPERATORS__")
                nvcc_flags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        
        return [flag for flag in nvcc_flags if flag]  # 移除空字符串
    
    def is_rocm_pytorch(self):
        """检查是否为ROCm版PyTorch"""
        _is_rocm_pytorch = False
        if int(torch.__version__.split('.')[0]) > 1 or (int(torch.__version__.split('.')[0]) == 1 and int(torch.__version__.split('.')[1]) >= 5):
            _is_rocm_pytorch = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if _is_rocm_pytorch:
                try:
                    from torch.utils.cpp_extension import ROCM_HOME
                    _is_rocm_pytorch = ROCM_HOME is not None
                except ImportError:
                    _is_rocm_pytorch = False
        return _is_rocm_pytorch
    
    def installed_rocm_version(self):
        """获取已安装的ROCm版本"""
        if not self.is_rocm_pytorch():
            return 0, 0
        
        ROCM_MAJOR = '0'
        ROCM_MINOR = '0'
        
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            rocm_ver_file = Path(ROCM_HOME).joinpath(".info/version")
            if rocm_ver_file.is_file():
                with open(rocm_ver_file, 'r') as file:
                    version_raw = file.read().strip()
                    version_parts = version_raw.split('.')
                    if len(version_parts) >= 2:
                        ROCM_MAJOR = version_parts[0]
                        ROCM_MINOR = version_parts[1]
            elif "rocm" in torch.__version__:
                rocm_ver_part = torch.__version__.split("rocm")[1]
                version_parts = rocm_ver_part.split('.')
                if len(version_parts) >= 2:
                    ROCM_MAJOR = version_parts[0]
                    ROCM_MINOR = version_parts[1]
            else:
                # 查找 /usr/include/rocm-version.h
                rocm_ver_file = Path("/usr/include/rocm_version.h")
                if rocm_ver_file.is_file():
                    with open(rocm_ver_file, 'r') as file:
                        for ln in file.readlines():
                            if "#define ROCM_VERSION_MAJOR" in ln:
                                ROCM_MAJOR = ln.strip().split()[-1]
                            elif "#define ROCM_VERSION_MINOR" in ln:
                                ROCM_MINOR = ln.strip().split()[-1]
        except Exception as e:
            print(f"警告: 无法检测ROCm版本: {e}")
        
        return int(ROCM_MAJOR), int(ROCM_MINOR)
    
    def get_rocm_gpu_arch(self):
        """获取ROCm GPU架构"""
        if not self.is_rocm_pytorch():
            return ""
        
        try:
            rocm_info = Path("/opt/rocm/bin/rocminfo")
            if not rocm_info.is_file():
                rocm_info = Path("rocminfo")
            
            rocm_gpu_arch_cmd = str(rocm_info) + " | grep -o -m 1 'gfx.*'"
            result = subprocess.check_output(rocm_gpu_arch_cmd, shell=True)
            rocm_gpu_arch = result.decode('utf-8').strip()
            return rocm_gpu_arch
        except subprocess.CalledProcessError:
            return ""
    
    def get_rocm_wavefront_size(self):
        """获取ROCm波前大小"""
        if not self.is_rocm_pytorch():
            return "32"
        
        try:
            rocm_info = Path("/opt/rocm/bin/rocminfo")
            if not rocm_info.is_file():
                rocm_info = Path("rocminfo")
            
            rocm_wavefront_size_cmd = str(rocm_info) + " | grep -Eo -m1 'Wavefront Size:[[:space:]]+[0-9]+' | grep -Eo '[0-9]+'"
            result = subprocess.check_output(rocm_wavefront_size_cmd, shell=True)
            rocm_wavefront_size = result.decode('utf-8').strip()
            return rocm_wavefront_size
        except subprocess.CalledProcessError:
            return "32"
    
    def load(self):
        """加载Fused Adam扩展"""
        # 首先尝试导入已安装的扩展
        try:
            return importlib.import_module("oom_optimizer._C.fused_adam")
        except (ImportError, ModuleNotFoundError):
            pass
        
        # 如果没有找到已安装的扩展，则JIT编译
        build_dir = self.get_extension_path()
        
        # 检查是否已经编译过
        jit_module_path = os.path.join(build_dir, "fused_adam.so")
        compiled_before = os.path.exists(jit_module_path)
        print(f"{'Loading' if compiled_before else 'Compiling'} FusedAdam optimizer...")
        
        start_time = time.time()
        
        # 为ROCm设置环境变量
        if self.is_rocm_pytorch():
            rocm_arch = self.get_rocm_gpu_arch()
            if rocm_arch:
                os.environ["PYTORCH_ROCM_ARCH"] = rocm_arch
        
        # 加载扩展
        op_module = load(
            name="fused_adam",
            sources=self.get_sources(),
            extra_include_paths=self.get_include_dirs(),
            extra_cflags=self.get_cxx_args(),
            extra_cuda_cflags=self.get_nvcc_args(),
            build_directory=build_dir,
        )
        
        duration = time.time() - start_time
        print(f"FusedAdam {'Load' if compiled_before else 'Compile'} finished, cost {duration:.2f} seconds")
        
        return op_module
