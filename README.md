# llm-table-extraction

`accelerate launch dummy_train.py`

There's something weird going on between nccl and torch version
Basically, installing torch 2.2.0 then reinstalling torch 2.1.0 seems to fix it
But don't really understand the logic