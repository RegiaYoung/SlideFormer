The [dataset_info.json](dataset_info.json) contains all available datasets. To use a custom dataset, add a dataset description in `dataset_info.json` and specify `dataset: name: your_dataset` in your training config.

Currently we support datasets in **alpaca** and **sharegpt** format. Allowed file types: JSON, JSONL.

```json
"dataset_name": {
  "hf_hub_url": "HuggingFace dataset repo (if specified, ignores file_name)",
  "file_name": "local file in this directory",
  "formatting": "alpaca (default) or sharegpt",
  "split": "dataset split (default: train)",
  "num_samples": "number of samples to use (optional)",
  "columns": {
    "prompt": "column name for user instruction (default: instruction)",
    "query": "column name for additional input (default: input)",
    "response": "column name for model response (default: output)",
    "system": "column name for system prompt (optional)",
    "messages": "column name for conversations (sharegpt, default: conversations)"
  },
  "tags": {
    "role_tag": "key for role in messages (default: from)",
    "content_tag": "key for content in messages (default: value)",
    "user_tag": "value representing user (default: human)",
    "assistant_tag": "value representing assistant (default: gpt)"
  }
}
```

## Alpaca Format

- [Example dataset](alpaca_en_demo.json)

The `instruction` column is the user prompt, `input` is optional additional context (concatenated to instruction), and `output` is the model response.

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "additional input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)"
  }
]
```

Dataset description in `dataset_info.json`:

```json
"my_dataset": {
  "file_name": "my_data.json"
}
```

## ShareGPT Format

Multi-turn conversations with role-based messages.

```json
[
  {
    "conversations": [
      {"from": "human", "value": "user instruction"},
      {"from": "gpt", "value": "model response"}
    ],
    "system": "system prompt (optional)"
  }
]
```

Dataset description in `dataset_info.json`:

```json
"my_dataset": {
  "file_name": "my_data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  }
}
```

### OpenAI Format

A special case of sharegpt with `role`/`content` keys:

```json
[
  {
    "messages": [
      {"role": "user", "content": "user instruction"},
      {"role": "assistant", "content": "model response"}
    ]
  }
]
```

```json
"my_dataset": {
  "file_name": "my_data.json",
  "formatting": "sharegpt",
  "columns": {"messages": "messages"},
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

## Using HuggingFace Datasets

Datasets hosted on the HuggingFace Hub can be loaded directly:

```json
"metamath": {
  "hf_hub_url": "meta-math/MetaMathQA",
  "columns": {
    "prompt": "query",
    "response": "response"
  }
}
```

## Using in Training Config

Reference a dataset by name in your YAML config:

```yaml
dataset:
  name: "alpaca_en_demo"
  dataset_dir: "data"
  max_seq_len: 1024
```

Or use a direct path (backward compatible):

```yaml
dataset:
  path: "/path/to/arrow/dataset"
  max_seq_len: 1024
  query_field: "query"
  response_field: "response"
```
