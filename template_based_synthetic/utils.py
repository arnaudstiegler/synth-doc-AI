import random
import string


def custom_metatype_fill(metatype: str) -> str:
    if "random_chars" in metatype:
        size = int(metatype.split("_")[-1])
        return "".join([random.choice(string.ascii_letters) for _ in range(size)])
    elif metatype == "id":
        return "".join([str(random.randint(0, 9)) for _ in range(9)])
    elif metatype == "sex":
        return random.choice(["M", "F"])
    else:
        raise ValueError(f"Custom metatype {metatype} not supported")
