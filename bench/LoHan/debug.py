param_names = {}


def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    # XXX: can probably make a map of param2module and vice-versa
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}
    
def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return "unknown"


def debug_param2name_id(param):
    return f"name={debug_param2name(param)} id={param.sb_id}"


def debug_param2name_id_shape(param):
    return f"name={debug_param2name(param)} id={param.sb_id} shape={param.data.shape}"


def debug_param2name_id_shape_device(param):
    return f"name={debug_param2name(param)} id={param.sb_id} shape={param.data.shape} device={param.device}"


def debug_param2name_id_numel(param):
    return f"name={debug_param2name(param)} id={param.sb_id} numel={param.numel()}"


def debug_param2name_id_shape_status(param):
    return f"name={debug_param2name(param)} id={param.sb_id} shape={param.data.shape} status={param.ds_status}"