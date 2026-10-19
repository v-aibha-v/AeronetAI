Dataset guide
The bundled dataset is small and deliberately imbalanced. It demonstrates the full pipeline and is not meant to measure real-world accuracy.

Layout:
dataset/raw/bird/ contains five bird images.
dataset/raw/drone/ contains fifty-five drone images.
dataset/manifest.json records the class counts and validation metadata.
dataset/validation_report.txt records the validation details.

Labels:
The images carry image-level class labels only. There are no bounding-box annotations. Custom object-detection training is not justified with this dataset.

Class distribution:
bird: 5 images
drone: 55 images
total: 60 source images

Duplicate handling:
A near-duplicate group was detected between two drone thumb files. The preparation pipeline groups nearly identical images so class splits stay clean.

Annotation previews:
Small annotation preview files are shipped with the source images. They contain no measurable bounding-box geometry and are not treated as labels.

