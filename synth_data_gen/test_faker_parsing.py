from decimal import Decimal
from datetime import datetime, date
import json
from synth_data_gen.utils import get_random_metatype


def format_element(element):
    """
    Convert an element to a string, applying custom formatting for specific types.
    """
    if isinstance(element, datetime):
        # Format datetime objects in a specific way
        return element.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(element, date):
        # Format date objects in a specific way
        return element.strftime("%Y-%m-%d")
    elif isinstance(element, Decimal):
        # Convert Decimal to a normalized string format, or use `format()` for custom formatting
        return str(element)
    elif isinstance(element, (list, set, tuple)):
        # Recursively format each element in the iterable
        return type(element)(format_element(e) for e in element)
    else:
        # Default to json.dumps for complex types or ensure correct handling of various data types like floats, ints
        return json.dumps(element)


def parse_iterable(iterable):
    """
    Parse an iterable, formatting each element as a string.
    """
    return [format_element(element) for element in iterable]


from faker import Faker

fake = Faker()

for metatype in metatypes:
    item = getattr(fake, metatype)()
    print(metatype, str(item))

    if len(str(item)) > 10000:
        print(metatype)
        assert False

    if isinstance(item, list):
        print(parse_iterable(item))
