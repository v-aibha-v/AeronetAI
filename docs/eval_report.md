Evaluation report schema
The evaluation pipeline writes a report to artifacts/evaluation.json. The report contains the dataset description, the split, and a metrics block for each model.


Top level:
- generated_at: when the report was created.- seed: the random seed used for the split.- positive_class: the class used as the positive class, if applicable.- dataset: the dataset description, including class distribution and a limitation note.- split: the training, validation, and test sizes and distributions.


Model metrics block:
Each model metrics block includes:
- architecture: a short description of the model.- available: whether the model produced predictions.- note: any warning when a model was unavailable.- precision, recall, f1, and support for each class, when available.- accuracy.- inference_time_ms.- uncertain_predictions and failed_predictions.


Honest metrics:
When a model is unavailable, its metrics are null. The report does not replace missing metrics with a generic value. It records that the metric is not available.

