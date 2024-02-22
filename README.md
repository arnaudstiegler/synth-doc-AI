# llm-table-extraction

`accelerate launch dummy_train.py`

There's something weird going on between nccl and torch version
Basically, installing torch 2.2.0 then reinstalling torch 2.1.0 seems to fix it
But don't really understand the logic


## benchmarks

With deepspeed Zero3.

All the numbers below are with max length 100

| Model   | Batch Size | GPU Ram (GB) |
|---------|------------|--------------|
| Phi-1.5 | 1          | 6            |
| Phi-2   | 1          | 9            |
| Phi-2   | 8          | 13           |
| Phi-2   | 16         | 18           |

Using a more realistic max length of 512

| Model   | Batch Size | GPU Ram (GB) |
|---------|------------|--------------|
| Phi-2   | 4          | 21       |
