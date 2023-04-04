## tasknet : simple multi-task transformers fine-tuning with Trainer and HuggingFace datasets. 
`tasknet` is an interface between Huggingface [datasets](https://huggingface.co/datasets) and Huggingface [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).


## Task templates
`tasknet` relies on task templates to avoid boilerplate codes. The task templates correspond to Transformers AutoClasses:
- `SequenceClassification` 
- `TokenClassification`
- `MultipleChoice`
- `Seq2SeqLM` (experimental support)

The task templates follow the same interface. They implement `preprocess_function`, a data collator and `compute_metrics`.
Look at [tasks.py](https://github.com/sileod/tasknet/blob/main/src/tasknet/tasks.py) and use existing templates as a starting point to implement a custom task template.

## Task instances and example

Each task template has fields that should be matched with specific dataset columns. Classification has two text fields `s1`,`s2`, and a label `y`. Pass a dataset to a template, and fill-in the mapping between the tempalte fields and the dataset columns to instanciate a task. 
```py
import tasknet as tn; from datasets import load_dataset

rte = tn.Classification(
    dataset=load_dataset("glue", "rte"),
    s1="sentence1", s2="sentence2", y="label") #s2 is optional

class hparams:
  model_name='microsoft/deberta-v3-base' # deberta models have the best results (and tasknet support)
  learning_rate = 3e-5 # see hf.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
 
tasks = [rte]
model = tn.Model(tasks, hparams)
trainer = tn.Trainer(model, tasks, hparams)
trainer.train()
trainer.evaluate()
p = trainer.pipeline()
p([{'text':x.premise,'text_pair': x.hypothesis}]) # HuggingFace pipeline for inference
```
Tasknet is multitask by design. `model.task_models_list` contains one model per task, with shared encoder.

## Installation
`pip install tasknet`

## Additional examples:
### Colab:
https://colab.research.google.com/drive/15Xf4Bgs3itUmok7XlAK6EEquNbvjD9BD?usp=sharing


## tasknet vs jiant
[jiant](https://github.com/nyu-mll/jiant/) is another library comparable to tasknet.  tasknet is a minimal extension of `Trainer` centered on task templates, while jiant builds a `Trainer` equivalent from scratch called [`runner`](https://github.com/nyu-mll/jiant/blob/master/jiant/proj/main/runner.py).
`tasknet` is leaner and closer to Huggingface native tools. Jiant is config-based and command line focused while tasknet is designed for interative use and python scripting.

## Credit

This code uses some part of the examples of the [transformers](https://github.com/huggingface/transformers/tree/main/src/transformers) library and some code from 
[multitask-learning-transformers](https://github.com/shahrukhx01/multitask-learning-transformers).

## Contact
You can request features on github or reach me at `damien.sileo@inria.fr`
```bib
@misc{sileod22-tasknet,
  author = {Sileo, Damien},
  doi = {10.5281/zenodo.561225781},
  month = {11},
  title = {{tasknet, multitask interface between Trainer and datasets}},
  url = {https://github.com/sileod/tasknet},
  version = {1.5.0},
  year = {2022}}
```
