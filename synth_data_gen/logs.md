## Entry 04/11:
- [x] Solving the font rendering
- [] Integrate back the default fonts

## Entry 04/10:
- [x] scale the structured box size with the font size
- [x] Make the table html more compact
- [x] Add the kv parsing as part of the preprocessing
- [x] Improve the font management

## Entry 04/09:
- [x] Fix the structured box html
- [x] Fix the checkpoint save strategy
- [] Make the table html more compact
- [] Add the kv parsing as part of the preprocessing
- [] scale the structured box size with the font size


## Entry 04/08:
- [x] Babysit the run
- [x] Add data aug. to make the model insensitive to key casing
- [x] Measure how often you end up with missing keys during training

Ideas:
- Add handwriting fonts
- Add other separators for k/v pairs
- Add structured box to the mix
- Lowercase / sensitivity to text key

## Entry 04/07:
- [x] Preprocess and generate a batch
- [x] Sanity check on created data
- [x] Training v0
- [] Think about next steps for the datagen

## Entry 04/04
### TODO:
- [x] Multiproc the datagen script
- [x] Add some more styling randomization
- [x] Cleanup the faker metatype generation
- [] Look into Genalog

## Entry 04/03:
### TODO:
- [x] Look into how to create a structured form
- [x] Think about how to randomize css styling more thoroughly

## Entry 04/02:
### TODO:
- [x] Figure out how to use a custom style -> Not sure if this is actually doable with pdfkit
- [x] Scrape fonts
- [x] Look into  ([weasyprint](https://github.com/Kozea/WeasyPrint?tab=BSD-3-Clause-1-ov-file#readme)) as an alternative
- [x] Update the main script to use weasyprint
- [x] Clean up the script
- [x] Find a way to randomize the layout entirely
- [x] Create a separate jinja env for components
- [x] random font in the css

## Entry 04/01
### TODO:
- [x] Integrate a random image into the html
- [x] Add back augraphy to see how it behaves
- [x] Update the column components to generate a random number of columns
- [x] Create helper to generate the kv_pairs
- [x] Fix PDF -> PNG conversion
- [x] Integrate the metatype into the variable name
- [] Figure out a solution for macro inputs
- [] Think about how to generate more style and templates easily -> Can we randomly sample jinja macros and add them?

## Entry 03/31
### TODO:
- [x] Find ways to collect all jinja attributes from the html (probably tricky)
- [x] Adapt the font size to the page? -> use em
- [x] Find a way to make the 2 columns work with the css
- [] Integrate the metatype into the variable name
- [x] Test out GPT to generate key-value pairs
- [] Figure out a solution for macro inputs
- [] Think about how to generate more style and templates easily


## Entry 03/27
### TODO:
- [x] Fix the bug on getting style.css
- [x] Try adding more "visual" elements to the page (randomizing the styling of the page?)
- [x] Use pdf instead of images
- [] Find ways to collect all jinja attributes from the html (probably tricky)
- [] Adapt the font size to the page?
- [] Find a way to make the 2 columns work with the css


## Entry 03/25
### TODO:
- [x] Fix the jinja macro issue for the table
- [x] Try creating a paragraph component
- [] Try adding more "visual" elements to the page (randomizing the styling of the page?)
- [] Fix the bug on getting style.css

## Entry 03/24

## TODO:
- Adjust box/text ratio
- Set max size for a box
- Add min/max size for a given component
- Try out paragraph
- Try out header

HTML format is much better for anything that's printed semi-structured and unstructured documents. Variability will have to come from modularizing different pieces from the 

For structured, it still makes sense to have the forms themselves rather than trying to recreate them (at least at first)

That doesn't cover things like IDs / Drivers licenses / checks etc... which are 

Multiple dimensions to this:
- number of different document layouts
- for a given layout, how many variation you can create (only talking about small changes to the layout, the text is easy to modulate)
- quality of the rendering (does this look like a document)
- diversity of the rendering (for a given template, how many variations can I create)

For a given layout:
- changing text
- changing fonts
- Varying layout

V0 draft:
- mostly about generating 