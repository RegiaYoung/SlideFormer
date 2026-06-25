import torch
def print_format(
    detail,
    opt_type = 'log', 
    loc = 'terminal',
    ):

    if loc == 'terminal':
        if opt_type == 'event':
            print(f'---------------------{detail}---------------------')
        elif opt_type == 'log':
            print(detail)
    else:
        if opt_type == 'event':
            print(f'---------------------{detail}---------------------', file=loc)
        elif opt_type == 'log':
            print(detail, file=loc)

def priority_sort(input_list):
    group_size = 4
    priority_order = [3, 1, 0, 2]
    sorted_list = []

    for priority in priority_order:
        for i in range(0, len(input_list), group_size):
            group = input_list[i:i + group_size]
            for element in group:
                if element % 4 == priority:
                    sorted_list.append(element)

    return sorted_list

def get_act_swap_list(fw_time, args, swap_list, act_pack, act_priority):
    act_idex = 0
    GB = 1024 * 1024 * 1024
    origin_chp_time = args.batch_size * args.max_seq_len * args.hidden_dim * 2 * 2 * args.num_layers / GB / 18
    swap_and_recompute_time = origin_chp_time
    count_size = 0
    if not args.is_fully_swap:
        while fw_time[0] * args.swap_ratio / 1000 > swap_and_recompute_time:
            now_act = act_priority[act_idex]
            if now_act % 4 == 3:
                count_size += 1
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 1:
                count_size += 1
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 0:
                count_size += 3
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 * 3 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim*3),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 2:
                count_size += 4
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 * 4 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim*4),dtype=torch.float16, pin_memory=True)
            
            act_idex += 1
            if act_idex == len(act_priority):
                break
    else:
        for act_idex in range(len(act_priority)):
            now_act = act_priority[act_idex]
            if now_act % 4 == 3:
                count_size += 1
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 1:
                count_size += 1
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 0:
                count_size += 3
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 * 3 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim*3),dtype=torch.float16, pin_memory=True)
            elif now_act % 4 == 2:
                count_size += 4
                swap_list.append(now_act)
                swap_and_recompute_time += args.batch_size * args.max_seq_len * args.hidden_dim * 2 * 4 / GB / 20
                act_pack[now_act] = torch.empty((args.batch_size*args.max_seq_len, args.hidden_dim*4),dtype=torch.float16, pin_memory=True)
            act_idex += 1
    print('swap_list: ', swap_list)