import os
import random

import matplotlib.font_manager


def find_ttf_files(folder_path):
    ttf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ttf"):
                ttf_files.append(os.path.abspath(os.path.join(root, file)))
    return ttf_files


def generate_css():
    def random_color():
        return f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"

    padding = random.randint(1, 10)
    font_size = random.randint(14, 18)
    margin_auto1 = random.randint(1, 3)
    margin_auto2 = random.randint(1, 3)
    border_value = "border: 1px solid black;" if random.random() < 0.1 else ""

    all_fonts = find_ttf_files(
        "synth_data_gen/google-fonts/"
    ) + matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    font = "file://" + random.choice(all_fonts)

    random_table_width = random.random()

    if random.random() < 0.4:
        # Weird format shouldn't account for too many cases
        width = random.randint(300, 500)
        height = random.randint(300, 500)
        page_size = f"@page {{size: {width}mm {height}mm ;}}"
    else:
        # Basically normal A4 size
        page_size = ""

    css_content = f"""
{page_size}
    
@font-face {{
font-family: 'Zeyada';
src: url({ font }) format('truetype');
}}

* {{
    font-family: 'Zeyada';
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

body {{ font-family: 'Zeyada'; font-size: {font_size}px; overflow: visible; margin: {margin_auto1}px {margin_auto2}px;}}
.header, .footer {{ text-align: center; font-family: 'Zeyada';}}
div {{ {border_value} font-family: 'Zeyada';}}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: {padding}px;}}
h3 {{ font-size: 1.5em;}}
    """
    return css_content
