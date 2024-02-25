import json
import os
import re

import click

from benchmark_agent.predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    Sample,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, "tasks_manual.json")))["samples"]

MODEL_CHOICES = {
    'mistral-orca': MistralOpenOrcaPredictor,
    'mistral-instruct': MistralInstructPredictor
}


def find_and_parse_json(s):
    # Regular expression pattern to find a JSON object
    # This pattern assumes the JSON does not contain nested objects or arrays
    pattern = r'\{[^{}]*\}'

    # Search for JSON string within the input
    match = re.search(pattern, s)

    # If a match is found, parse the JSON string
    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            return None
    return None


@click.command()
@click.option('--model', type=click.Choice(list(MODEL_CHOICES.keys())), default='mistral-instruct')
def run_eval(model: str) -> None:
    model_predictor_cls = MODEL_CHOICES[model]
    model_predictor = model_predictor_cls()

    valid_json = []
    valid_plain = []
    for entry in samples:
        sample = Sample(**entry)
        generated_answer = model_predictor.generate_answer(sample=sample)

        if sample.json_expected:
            try:
                json_answer = json.loads(generated_answer)
                valid_json.append(1)
            except:
                out = find_and_parse_json(generated_answer)
                if out:
                    # TODO: invalid format a priori, but could be recovered?
                    valid_json.append(0)
                else:
                    valid_json.append(0)
        else:
            # TODO: tbd whether we want to do this
            if generated_answer == sample.expected_output:
                valid_plain.append(1)
            elif sample.expected_output in generated_answer:
                valid_plain.append(0)
            else:
                valid_plain.append(0)


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
