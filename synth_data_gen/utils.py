import re


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def parse_jinja_variables(html_path: str):
    '''
    Warning: this is pretty brittle. Make sure all variables are formatted as "{{ variable }}"
    '''
    with open(html_path, 'r') as file:  # r to open file in READ mode
        html_as_string = file.read()
        return re.findall("\{\{\s(.*?)\s\}\}", html_as_string)