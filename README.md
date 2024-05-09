# Synthetic Document Generation

Real-world document data is hard to come by: documents usually contain plenty of sensitive PII (Personal Identifiable Information), and they are hard to collect outside of a corporate structure.
This repo aims at addressing that gap by generating high-quality synthetic documents that can be used as a proxy or a complement to any document extraction task.

## Two Pipelines:

### Template-based Synthetic data

Some well-known document types (like passports and drivers licenses) are ubiquitous in document processing, but hard to come by because
they are filled with PII information. This pipeline generates synthetic documents for those templated document types.

Some pointers:
- Works with "empty" templates that gets populated and pasted into a document
- Can be extended to new document types
- Currently only supports passports
- See HF datasets

Future work:
- Extend to more document types
- More refinement on the randomized generation pipeline



### HTML-based synthetic data

This pipeline generates "semi-structured" documents using HTML/PDF: it aims at reproducing layouts that you would encounter for documents like invoices (so not entirely unstructured). The document structure is fully randomized and offers a lot of variability.
