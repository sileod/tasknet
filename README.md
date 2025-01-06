## tasknet : simple modernBERT fine-tuning, with multi-task support
`tasknet` is an interface between Huggingface [datasets](https://huggingface.co/datasets) and Huggingface transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

Tasknet should work with all recent versions of Transformers.

## Installation and example

`pip install tasknet`

Each task template has fields that should be matched with specific dataset columns. Classification has two text fields `s1`,`s2`, and a label `y`. Pass a dataset to a template, and fill in the mapping between the template fields and the dataset columns to instantiate a task. 
```py
import tasknet as tn; from datasets import load_dataset

rte = tn.Classification(
    dataset=load_dataset("glue", "rte"),
    s1="sentence1", s2="sentence2", y="label") #s2 is optional for classification, used to represent text pairs
 # See AutoTask for shorter code

class hparams:
  model_name = 'tasksource/ModernBERT-base-nli' # better performance for most tasks
  learning_rate = 3e-5 # see hf.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
 
model, trainer = tn.Model_Trainer(tasks=[rte],hparams)
trainer.train(), trainer.evaluate()
p = trainer.pipeline()
p([{'text':'premise here','text_pair': 'hypothesis here'}]) # HuggingFace pipeline for inference
```
Tasknet is multitask by design. `model.task_models_list` contains one model per task, with a shared encoder.

## Task templates
`tasknet` relies on task templates to avoid boilerplate codes. The task templates correspond to Transformers AutoClasses:
- `SequenceClassification` 
- `TokenClassification`
- `MultipleChoice`
- `Seq2SeqLM` (experimental support)

The task templates follow the same interface. They implement `preprocess_function`, a data collator and `compute_metrics`.
Look at [tasks.py](https://github.com/sileod/tasknet/blob/main/src/tasknet/tasks.py) and use existing templates as a starting point to implement a custom task template.


## AutoTask
You can also leverage [tasksource](https://github.com/sileod/tasksource/) with tn.AutoTask and have one-line access to 600+ datasets, see [implemented tasks](https://github.com/sileod/tasksource/blob/main/README.md).
```py
rte = tn.AutoTask("glue/rte", nrows=5000)
```
AutoTask guesses a template based on the dataset structure. It also accepts a dataset as input, if it fits the template (e.g. after tasksource custom preprocessing).

## Balancing dataset sizes 
```py
tn.Classification(dataset, nrows=5000, nrows_eval=500, oversampling=2)
```
You can balance multiple datasets with `nrows` and `oversampling`. `nrows` is the maximal number of examples. If a dataset has less than `nrows`, it will be oversampled at most `oversampling` times.


## Colab examples
Minimal-ish example:

https://colab.research.google.com/drive/15Xf4Bgs3itUmok7XlAK6EEquNbvjD9BD?usp=sharing

More complex example, where tasknet was scaled to [600 tasks](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli):

https://colab.research.google.com/drive/1iB4Oxl9_B5W3ZDzXoWJN-olUbqLBxgQS?usp=sharing

## tasknet vs jiant
[jiant](https://github.com/nyu-mll/jiant/) is another library comparable to tasknet.  tasknet is a minimal extension of `Trainer` centered on task templates, while jiant builds a `Trainer` equivalent from scratch called [`runner`](https://github.com/nyu-mll/jiant/blob/master/jiant/proj/main/runner.py).
`tasknet` is leaner and closer to Huggingface native tools. Jiant is config-based and command line focused while tasknet is designed for interactive use and python scripting.

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
