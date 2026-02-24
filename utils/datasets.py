import json
import torch
from torch.utils.data import Dataset


class MathFusionQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=4096):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        
        # 分别处理问题和答案部分
        question = f"Question: {example['query']}\nAnswer:"
        answer = example['response']
        
        # 编码问题部分以获取其长度（不加特殊token以匹配拼接后的编码）
        question_ids = self.tokenizer(
            question, 
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        question_len = question_ids.shape[0]
        
        # 编码完整文本
        full_text = question + " " + answer
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # 创建标签
        labels = input_ids.clone()
        
        # 1. 将padding位置的labels设为-100
        labels[attention_mask == 0] = -100
        
        # 2. 将问题部分设为-100（忽略），只对答案部分计算损失
        bos_shift = 1 if self.tokenizer.bos_token_id is not None else 0
        labels[:question_len + bos_shift] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class AlpacaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inst_close_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        # print(f"[DEBUG] self.inst_close_id 类型: {type(self.inst_close_id)}")  # 应为 int
        # print(f"[DEBUG] self.inst_close_id 值: {self.inst_close_id}")          # 应为有效整数（如 128001）
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 构建提示模板
        prompt = f"<|start_header_id|>ststem<|end_header_id|>\n\n{item['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{item['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}" 
        
        encodings = self.tokenizer(
            prompt, 
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            add_special_tokens=True
        )
        # Get the actual token length before padding
        # actual_token_length = encodings['attention_mask'].sum().item()
        # print(f"[DEBUG] actual_token_length: {actual_token_length}")
        # 获取 input_ids 和 attention_mask
        input_ids = encodings['input_ids'].squeeze(0)
        # print(f"[DEBUG] input_ids 类型: {input_ids.dtype}") 
        attention_mask = encodings['attention_mask'].squeeze(0)
        # print(f"[DEBUG] attention_mask 类型: {attention_mask.dtype}")
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # inst_close_idx = (input_ids == self.inst_close_id).nonzero(as_tuple=True)[0]
        # # print(inst_close_idx)
        # if inst_close_idx.numel() > 0:
        #     labels[:inst_close_idx[-1]+1] = -100
        # else:
        #     labels[:] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
class DummyDataset(Dataset):
    def __init__(self, size=1000, tokenizer=None, max_length=512):
        """
        Initialize a dummy dataset that generates random data for Llama 3.1 fine-tuning.
        
        Args:
            size (int): Number of samples in the dataset
            tokenizer: The tokenizer (optional, only used for vocab size)
            max_length (int): Maximum sequence length
        """
        self.size = size
        self.max_length = max_length
        self.vocab_size = 32000  # Default for Llama models
        
        if tokenizer is not None:
            self.vocab_size = tokenizer.vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random input_ids
        input_ids = torch.randint(1, self.vocab_size, (self.max_length,))
        
        # Generate random attention_mask (mostly 1s with some 0s at the end)
        # Ensure it's not all ones by capping seq_length to be less than max_length
        seq_length = torch.randint(self.max_length // 3, self.max_length - 1, (1,)).item()
        
        # Here use bool for attention_mask (maybe only for PyTorch 2.0+ or flash_attn)
        attention_mask = torch.zeros(self.max_length, dtype=torch.bool)
        attention_mask[:seq_length] = 1
        # attention_mask = torch.ones(self.max_length, dtype=torch.long)
        # attention_mask[self.max_length - 1] = 0
        
        # Create labels - simulate instruction following format
        labels = input_ids.clone()
        
        # Set labels to -100 where attention_mask is 0
        labels[attention_mask == 0] = -100
        
        # Simulate instruction part with -100 labels for first portion
        instruction_length = torch.randint(10, min(100, self.max_length // 5), (1,)).item()
        labels[:instruction_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }