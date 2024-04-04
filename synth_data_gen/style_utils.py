import random
import matplotlib.font_manager


def generate_css():
    def random_color():
        return f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

    padding = random.randint(1, 50)  # Random padding between 5px and 10px
    font_size = random.uniform(0.5, 1.0)  # Random font size between 0.8em and 1.2em
    margin_auto = random.randint(5, 50)  # Random margin between 400px and 600px
    border_value = 'border: 1px solid black;' if random.random() < 1/3.0 else ''
    maybe_flex = 'flex:1 ;' if random.random() < 1/3.0 else ''
    font = 'file://' + random.choice(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
    
    css_content = f"""
@font-face {{
font-family: test;
src: url({font}) ;
}}

.table {{
    font-family: Arial, sans-serif;
    width: 80%;
    border-collapse: collapse;
}}

.table table {{
    width: 100%;
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

body {{ font-family: 'test'; font-size: {font_size}em; margin: {margin_auto}px auto; text; { maybe_flex }}}
.header, .footer {{ text-align: center; }}
div {{ {border_value} {maybe_flex} }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: {padding}px; font-size: {font_size}em; }}
h3 {{ font-size: 1.5em; {maybe_flex}}}
    """
    return css_content
