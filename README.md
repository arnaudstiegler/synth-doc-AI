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


When using LORA:
- GPU RAM usage is very close (~1gb gap)
- CPU usage is <<<
- training step is about 3 times faster
This can be explained by the optimizer state CPU offloading: we reduce the size of the optimizer state that is lopcated on the CPU. So there's little impact on the GPU ram usage

With 4bit quantization: 7.5GB of GPU Ram usage


Deepspeed Accelerate and quantization: 1.3s per step