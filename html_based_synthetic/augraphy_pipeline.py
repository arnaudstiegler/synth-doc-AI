import random
from augraphy import *


ink_phase = [
    OneOf(
        [
    InkBleed(
        intensity_range=(0.4, 0.7),
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
    ),
    InkShifter(
    ),
    BleedThrough(
    ),
], p=1.0)
        ]

paper_phase = [
            OneOf(
        [
            InkBleed(
        intensity_range=(0.4, 0.7),
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
    ),
            DelaunayTessellation(
                n_points_range=(500, 800),
                n_horizontal_points_range=(500, 800),
                n_vertical_points_range=(500, 800),
                noise_type="random",
                color_list="default",
                color_list_alternate="default",
            ),
            PatternGenerator(
                imgx=random.randint(256, 512),
                imgy=random.randint(256, 512),
                n_rotation_range=(10, 15),
                color="random",
                alpha_range=(0.25, 0.5),
            ),
            VoronoiTessellation(
                mult_range=(50, 80),
                seed=19829813472,
                num_cells_range=(500, 1000),
                noise_type="random",
                background_value=(200, 255),
            ),
            ColorPaper(
                hue_range=(0, 255),
                saturation_range=(10, 40),
            ),
            NoiseTexturize(
                    sigma_range=(3, 10),
                    turbulence_range=(2, 5),
                ),
            BrightnessTexturize(
                    texturize_range=(0.9, 0.99),
                    deviation=0.03,
                ),
        ],
        p=1.0,
    ),
]

post_phase = [
     OneOf(
        [
            SubtleNoise(
                    subtle_range=random.randint(5, 10),
                ),
                Jpeg(
                    quality_range=(25, 95),
                ),
            Markup(
                num_lines_range=(2, 7),
                markup_length_range=(0.5, 1),
                markup_thickness_range=(1, 2),
                markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
                markup_color="random",
                single_word_mode=False,
                repetitions=1,
            ),
            Scribbles(
                scribbles_type="random",
                scribbles_location="random",
                scribbles_size_range=(250, 600),
                scribbles_count_range=(1, 6),
                scribbles_thickness_range=(1, 3),
                scribbles_brightness_change=[32, 64, 128],
                scribbles_text="random",
                scribbles_text_font="random",
                scribbles_text_rotate_range=(0, 360),
                scribbles_lines_stroke_count_range=(1, 6),
            ),
            DirtyDrum(
                line_width_range=(1, 6),
                line_concentration=random.uniform(0.05, 0.15),
                direction=random.randint(0, 2),
                noise_intensity=random.uniform(0.6, 0.95),
                noise_value=(64, 224),
                ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
                sigmaX=0,
                p=0.2,
            ),
            DirtyRollers(
                line_width_range=(2, 32),
                scanline_type=0,
            ),
            ColorShift(
                color_shift_offset_x_range=(3, 5),
                color_shift_offset_y_range=(3, 5),
                color_shift_iterations=(2, 3),
                color_shift_brightness_range=(0.9, 1.1),
                color_shift_gaussian_kernel_range=(3, 3),
            ),
            BadPhotoCopy(
                noise_type=-1,
                noise_side="random",
                noise_iteration=(1, 2),
                noise_size=(1, 3),
                noise_value=(128, 196),
                noise_sparsity=(0.3, 0.6),
                noise_concentration=(0.1, 0.6),
                blur_noise=random.choice([True, False]),
                blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
                wave_pattern=random.choice([True, False]),
                edge_effect=random.choice([True, False]),
            ),
        
                    Faxify(
                scale_range=(0.1, 0.2),
                monochrome=random.choice([0, 1]),
                monochrome_method="random",
                monochrome_arguments={},
                halftone=random.choice([0, 1]),
                invert=1,
                half_kernel_size=random.choice([(1, 1), (2, 2)]),
                angle=(0, 360),
                sigma=(1, 5),
            ),
        ],
        p=1.0,
    ),
]

AUG_PIPE = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)