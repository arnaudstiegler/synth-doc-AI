import json
import os
from benchmark_agent.predictors import MistralOpenOrcaPredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, 'tasks.json')))['samples']

model_predictor = MistralOpenOrcaPredictor()

for sample in samples:
    prompt = sample['task_definition'] + ' ' + sample['task_input']
    expected_answer = sample['expected_output']
    generated_answer = model_predictor.generate_answer(prompt)
    print(generated_answer, expected_answer)