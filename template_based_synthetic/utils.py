import random
import string
from datetime import datetime
import json
import re
from typing import Optional


def custom_metatype_fill(metatype: str) -> str:
    if "random_chars" in metatype:
        size = int(metatype.split("_")[-1])
        return "".join([random.choice(string.ascii_letters) for _ in range(size)])
    elif metatype == "id":
        return "".join([str(random.randint(0, 9)) for _ in range(9)])
    elif metatype == "sex":
        return random.choice(["M", "F"])
    elif metatype == "fixed_usa":
        return "USA"
    else:
        raise ValueError(f"Custom metatype {metatype} not supported")


def format_date(date: str) -> str:
    parsed_date = datetime.strptime(date, "%Y-%m-%d")
    return parsed_date.strftime("%d %b %Y")


def extract_and_parse_json(input_string: str) -> Optional[str]:
    pattern = r"\{(.*)\}"
    match = re.search(pattern, input_string, re.S)
    if match:
        try:
            json_content = "{" + match.group(1) + "}"
            parsed_json = json.loads(json_content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None
    return None
