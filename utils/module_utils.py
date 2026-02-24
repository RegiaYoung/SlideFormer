

def _print_module_structure(module, depth=0, prefix=''):
    """递归打印模块结构"""
    indent = '  ' * depth
    module_name = module.__class__.__name__
    
    # 获取模块的设备信息
    device_info = ''
    if hasattr(module, 'device'):
        device_info = f" (device: {module.device})"
    elif next(module.parameters(), None) is not None:
        device_info = f" (device: {next(module.parameters()).device})"
        
    # 获取模块的数据类型信息
    dtype_info = ''
    if hasattr(module, 'dtype'):
        dtype_info = f" (dtype: {module.dtype})"
    elif next(module.parameters(), None) is not None:
        dtype_info = f" (dtype: {next(module.parameters()).dtype})"
        
    print(f"{indent}{prefix}{module_name}{device_info}{dtype_info}")

    # 打印参数数量
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if total_params > 0:
        print(f"{indent}  Parameters: {total_params:,}")

    # 递归处理子模块
    for name, child in module.named_children():
        _print_module_structure(child, depth + 1, f"{name}: ")