import json
import os
from benchmark_agent.predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    Sample,
)
import click

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, "tasks_manual.json")))["samples"]

MODEL_CHOICES = {
    'mistral-orca': MistralOpenOrcaPredictor,
    'mistral-instruct': MistralInstructPredictor
}

@click.command()
@click.option('--model', type=click.Choice(list(MODEL_CHOICES.keys())), default='mistral-instruct')
def run_eval(model:str) -> None:
    model_predictor_cls = MODEL_CHOICES[model]
    model_predictor = model_predictor_cls()

    valid_json = []
    for sample in samples:
        generated_answer = model_predictor.generate_answer(sample=Sample(**sample))
        try:
            json_answer = json.loads(generated_answer)
            valid_json.append(1)
        except:
            valid_json.append(0)

        '''
        Use jsonformer to force json output
        For eval:
        if json_expected: try json.loads first. If not working, try retrieve a json from the output and load it
        Verify the keys
        if not json_expected: try to match only the output, else try to find the answer in the output
        '''

    print(sum(valid_json) / len(valid_json))


if __name__ == '__main__':
    run_eval()