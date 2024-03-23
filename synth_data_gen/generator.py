"""

Structured:
- Cells with key in it

Semi-structured:
- key-values pairs, tables, random shortish text, header

Unstructured:
- paragraphs
- header

"""
from PIL import Image, ImageDraw
import random
import string
from synth_data_gen.text_box import KeyValueBox


A4_aspect_ratio = 16 / 9
Base_width = 3200


def divide_rectangle(x, y, width, height, depth=0):
    rectangles = []  # Stores final rectangles with coordinates
    
    # Decide randomly not to divide further or based on depth
    actions = ['x', 'y']

    if width < 1000 or height < 2000:
        actions.append('stop')
    elif width < 300 or height < 100:
        return [(int(x), int(y), int(width), int(height))]

    action = random.choice(actions)
    
    if action == 'stop' or depth == 10:  # Depth limit for recursion termination
        return [(int(x), int(y), int(width), int(height))]
    
    # Recursive case: divide and recurse
    if action == 'x':
        # Divide width by a random factor
        factor = random.uniform(0.3, 0.7)
        width1 = width * factor
        width2 = width - width1
        
        # Recurse for each new rectangle, updating x for the second rectangle
        rectangles += divide_rectangle(x, y, width1, height, depth+1)
        rectangles += divide_rectangle(x + width1, y, width2, height, depth+1)
        
    elif action == 'y':
        # Divide height by a random factor
        factor = random.uniform(0.3, 0.7)
        height1 = height * factor
        height2 = height - height1
        
        # Recurse for each new rectangle, updating y for the second rectangle
        rectangles += divide_rectangle(x, y, width, height1, depth+1)
        rectangles += divide_rectangle(x, y + height1, width, height2, depth+1)
        
    return rectangles

class DocumentGenerator:
    def create():
        base_image = Image.new("RGBA", (int(Base_width), int(Base_width*A4_aspect_ratio)), color="white")

        rectangles = divide_rectangle(0, 0, int(Base_width), int(Base_width*A4_aspect_ratio))

        for rectangle in rectangles:
            x_position, y_position, rect_width, rect_height = rectangle
            new_component = KeyValueBox.generate(int(rect_height), int(rect_width))
            base_image.paste(new_component, (x_position, y_position, x_position + new_component.size[0], y_position + new_component.size[1]))

        return base_image


if __name__ == "__main__":
    img = DocumentGenerator.create()
    img.show()
