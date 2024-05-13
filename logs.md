## 05/10:
- [x] Create HF datasets
- [x] Publish them on the hub
- [] List out next steps

## 05/03:
- [x] Dynamic size for the template
- [x] Random rotation on pasting
- [x] Random background image
- [x] Multiproc

## 04/28:
- [x] Added metatype support for template-based
- [x] Annotated the passport templates

https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download

Next:
- Add the right font
- maybe fix the bboxes
- test out second template
- return sample metadata
- Augraphy

## 04/27:
- [x] Missed a bunch of entries here (including batch 4 with no missing values and a segmentation version)
- [x] Started folder for template-based synthetic data

## 04/22:
- [x] Sanity check on the batch 3
- [x] Play around with accelerate to see whether this could be better than the HF trainer
- [x] Start run on batch 3

Note: the augmentation from Augraphy is a bit too strong, makes most page really hard to read.

## Entry 04/21:
- [x] Adding one line kv html component
- [x] sanity check on the generated samples
- [x] investigate missing kv pairs -> caused by overflow
- [x] cleanup overlap between some of the components
- [x] Adding variable page size
- [x] Restart generation

## Entry 04/18:
- [x] Run some tests on the trained model: performing worse, often returns "Missing" on fields that are present
- [x] Check presence rate for old vs new data
- [x] Should add a stricter count for documents with no kv-pairs?
- [x] Look into documents that have no kv pairs to make sure they're not hiding a pattern

## Entry 04/15:
- [x] Understand why torch.compile doesn't work
- [x] Debug the run

## Entry 04/13:
- [x] Started run with v2
- [] Why is the gradient norm >> 1 while max_grad_norm=1 in the trainer arguments?

## Entry 04/11:
- [x] Solving the font rendering
- [x] Integrate back the default fonts
- [x] Start generation second batch
- [] Clean up the codebase

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