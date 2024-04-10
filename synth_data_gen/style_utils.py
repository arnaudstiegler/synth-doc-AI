import random
import matplotlib.font_manager


def generate_css():
    def random_color():
        return f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

    padding = random.randint(1, 10)
    font_size = random.uniform(10, 20)
    # margin_auto = random.randint(5, 20)
    margin_auto = 1
    border_value = "border: 1px solid black;" if random.random() < 0.1 else ""
    font = "file://" + random.choice(
        matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    )
    random_table_width = random.random()

    css_content = f"""
@font-face {{
font-family: random_font;
src: url({font}) format('truetype');
}}

* {{
    font-family: 'random_font';
}}

.table {{
    width: {int(random_table_width*100)}%;
    border-collapse: collapse;
}}

.table th, .table td {{
    border: 1px solid #ddd;
    padding: {padding}px;
    text-align: left;
}}

.table th {{
    background-color: {random_color()};
}}

.table tfoot td {{
    font-weight: bold;
}}

table tbody tr:nth-child(even){{background-color: {random_color()};}}

body {{ font-family: 'random_font'; font-size: {font_size}px; margin: {margin_auto}px auto; overflow: visible;}}
.header, .footer {{ text-align: center; }}
div {{ {border_value} }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: {padding}px;}}
h3 {{ font-size: 1.5em;}}
    """
    return css_content
