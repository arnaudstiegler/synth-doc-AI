from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# @font-face {
#             font-family: 'custom';
#             src: url('/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/fonts/bhel_puri/Bhel Puri.otf') format('opentype'),
#         }

font_config = FontConfiguration()
html = HTML(string='<h1>The title</h1>')
css = CSS(string='''
    @font-face {
        font-family: test;
        src: url('/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/fonts/bhel_puri/Bhel Puri.otf');
    }
    h1 { font-family: test }''', font_config=font_config)
html.write_pdf(
    '/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/samples/sample_0.pdf', stylesheets=[css],
    font_config=font_config)