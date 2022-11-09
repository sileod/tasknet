# tasknet
`tasknet` is an interface between Huggingface [datasets](https://huggingface.co/datasets) and Huggingface [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).


## Task templates
`tasknet` relies on task templates to avoid boilerplate codes. The task templates are correspond to Transformers AutoClasses:
- `SequenceClassification` 
- `TokenClassification`
- `MultipleChoice`

The task templates follow an identical structure. They implement `preprocess_function` and `compute_metrics`.
Look at [tasks.py](https://github.com/sileod/tasknet/blob/main/src/tasknet/tasks.py) and use existing templates as a starting point to implement a custom task template.

## Instanciating a task

Each task is associated with specific fields. Classification has two text fields `s1`,`s2`, and a label `y`. pass a dataset to a template, and fill-in the mapping between the dataset fields and the template fields. 
```py
import tasknet as tn
from datasets import load_dataset

rte = tn.Classification(
    dataset=load_dataset("glue", "rte"),
    s1="sentence1", s2="sentence2", y="label"
)

class args:
  model_name='roberta-base'
  learning_rate = 3e-5 # see https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments

 
tasks = [rte]
models = tn.Model(tasks, args)
trainer = tn.Trainer(models, tasks, args)
trainer.train()
```
As you can see, tasknet is multitask by design. It works with list of tasks and list of models.

## Installation
`pip install tasknet`

## Additional examples:
https://colab.research.google.com/drive/15Xf4Bgs3itUmok7XlAK6EEquNbvjD9BD?usp=sharing

## tasknet vs jiant
[jiant](https://github.com/nyu-mll/jiant/tree/master/jiant) is another library comparable to tasknet.  tasknet is a minimal extension of `Trainer` centered on task templates, while jiant builds a custom analog of `Trainer` from scratch called [`runner`](https://github.com/nyu-mll/jiant/blob/master/jiant/proj/main/runner.py).
`tasknet` is leaner and easier to extend. jiant is config-based while tasknet is designed for interative use and scripting.

## Contact
You can request features on github or reach me at `damien.sileo@inria.fr`
