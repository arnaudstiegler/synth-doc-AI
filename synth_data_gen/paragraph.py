from PIL import Image, ImageDraw, ImageFont
import textwrap
from synth_data_gen.faker_utils import RealisticDataGenerator


PRINTED_FONT_PATH = "/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/fonts/Josefin_Sans/JosefinSans-VariableFont_wght.ttf"
HANDWRITTEN_FONT_PATH = "/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/fonts/Reenie_Beanie/ReenieBeanie-Regular.ttf"


data_gen = RealisticDataGenerator()


class Paragraph:
    MIN_HEIGHT = 300
    MAX_HEIGHT = 1000
    MIN_WIDTH = 500
    MAX_WIDTH = 1e9 # No max width
    TEXT_BOX_RATIO = 0.1

    def is_viable(self, height: int, width: int):
        if (self.MIN_HEIGHT <= height <= self.MAX_HEIGHT) and (self.MIN_WIDTH <= width <= self.MAX_WIDTH):
            return True
        return False

    def generate(self, height: int, width: int):

        text_image_size = (width, height)
        text_image = Image.new(
            "RGBA", text_image_size, (255, 255, 255, 255)
        )

        draw = ImageDraw.Draw(text_image)

        draw.rectangle(
            [(0, 0), (text_image_size[0] - 4, text_image_size[1] - 4)], outline="black"
        )


        font = ImageFont.truetype(
            PRINTED_FONT_PATH,
            size=width*self.TEXT_BOX_RATIO,
        )
        second_font = ImageFont.truetype(
            HANDWRITTEN_FONT_PATH,
            size=width*self.TEXT_BOX_RATIO,
        )

        # Draw the text (without specifying font will use default)
        draw.text((5, 5), "First Name", fill="black", font=font)  # , font=font

        # TODO: update the datagen call
        wrapped_text = textwrap.fill(data_gen.generate_data()["first_name"], width=text_image_size[1])
        draw.text(
            (text_image_size[0] // 4, text_image_size[1] // 2), wrapped_text, fill="black", font=second_font
        )

        return text_image


if __name__ == "__main__":
    kv_box, _ = KeyValueBox.generate(height=100, width=200)
    kv_box.show()

