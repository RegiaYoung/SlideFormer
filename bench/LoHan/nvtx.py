import torch
counters = {}
def nvtx_wrap(func):

    counters[func.__qualname__] = 0
    def wrapped_fn(*args, **kwargs):

        global counters
        
        counters[func.__qualname__] += 1
        name = func.__qualname__ + str(counters[func.__qualname__])

        torch.cuda.nvtx.range_push(name)
        ret_val = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret_val

    return wrapped_fn