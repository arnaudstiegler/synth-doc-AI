import json
import os
from benchmark_agent.predictors import MistralOpenOrcaPredictor

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, 'correctly_updated_tasks.json')))['samples']

model_predictor = MistralOpenOrcaPredictor()

valid_json = []
for sample in samples:
    prompt = sample['task_definition'] + ' ' + sample['task_input']
    expected_answer = sample['expected_output']
    generated_answer = model_predictor.generate_answer(prompt)
    print(generated_answer, expected_answer)

    # TODO: compute 1) is json?, 2) is right keys?, 3) is right answer?

    try:
        json_answer = json.loads(generated_answer)
        valid_json.append(1)
    except:
        valid_json.append(0)

print(sum(valid_json) / len(valid_json))