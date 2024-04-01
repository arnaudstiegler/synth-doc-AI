## Entry 04/01
### TODO:
- [x] Integrate a random image into the html
- [x] Add back augraphy to see how it behaves
- [x] Update the column components to generate a random number of columns
- [x] Create helper to generate the kv_pairs
- [] Fix PDF -> PNG conversion
- [x] Integrate the metatype into the variable name
- [] Figure out a solution for macro inputs
- [] Think about how to generate more style and templates easily

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