from dataclasses import dataclass
import numpy as np
import funcy as fc
from datasets import Dataset, DatasetDict, get_dataset_config_names,load_dataset
import pandas as pd
from collections import defaultdict

split_mapping={
'train':['train_split','training'],
'validation':['validation','dev','eval_split']
}

fields_mapping={
    'sentence1':['premise','sentence','sentence1','text','head','question1','inputs','question','sentence_A','Sentence','content','sms','tweet'],
    'sentence2':['hypothesis','sentence2','tail','question2','sentence_B'],
    'labels':['label','labels','relation','gold_label','Label','class', 'score','rating','star','meanGrade']
}

categories_mapping={'text-classification':'Classification',
'token-classification':'TokenClassification',
 'multiple-choice':'MultipleChoice'
}

def make_t0_mapping():
    df=pd.read_csv('https://raw.githubusercontent.com/bigscience-workshop/t-zero/master/t0/datasets.csv')

    task_type={'QA_multiple_choice':'MultipleChoice',
    'story_completion':'MultipleChoice',
    'coreference':'MultipleChoice',
    'summarization':'ConditionalGeneration',
    'structure_to_text':'ConditionalGeneration',
    'QA_closed_book':'ConditionalGeneration',
    'sentiment':'Classification',
    'topic_classification':'Classification',
    'NLI':'Classification',
    'paraphrase':'Classification',
    'QA_extractive':'QuestionAnswering'
    }

    df['task_type']=df.task_by_convention.map(lambda x:task_type.get(x,None))
    df['config']=df['subset']
    df['dataset_name']=df['HF_name']
    tuple_to_task_type=df.set_index(['HF_name','config'])['task_type'].to_dict()
    tuple_to_task_type[('super_glue','axg')]='Classification'
    tuple_to_task_type[('super_glue','wic')]='Classification'
    tuple_to_task_type[('super_glue','wic.fixed')]='Classification'
    tuple_to_task_type[('super_glue','boolq')]='Classification'
    tuple_to_task_type[('super_glue','multirc')]='Classification'

    return tuple_to_task_type

t0_tuple_to_task_type=make_t0_mapping()
def t0_task_type(x):

    config = x.config if x.config else None
    return t0_tuple_to_task_type.get((x.dataset_name, config),None)

def align_splits(dataset):
    for k,v in split_mapping.items():
        bad_name=[x for x in v if x in dataset and x!=k]
        if bad_name:
            dataset[k]=dataset[bad_name[0]]
            del dataset[bad_name[0]]
    return dataset

def fix_splits(dataset):
    if 'auxiliary_train' in dataset:
        del dataset['auxiliary_train']
    
    if 'test' in dataset:
        if 'label' in dataset['test'].features:
            if len(set(dataset['test'].to_dict()['label']))==1:
                 # obfuscated label
                del dataset['test']
    
    if 'validation' in dataset and 'test' not in dataset:
        validation_test = dataset['validation'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'test' in dataset and 'validation' not in dataset:
        validation_test = dataset['test'].train_test_split(0.5, seed=0)
        dataset['validation'] = validation_test['train']
        dataset['test']=validation_test['test']

    if 'validation' not in dataset and 'test' not in dataset:
        train_val_test = dataset["train"].train_test_split(seed=0)
        val_test = train_val_test["test"].train_test_split(0.5, seed=0)
        dataset["train"] = train_val_test["train"]
        dataset["validation"] = val_test["train"]
        dataset["test"] = val_test["test"]
        
    return dataset



def align_fields(dataset):
    for k,v in fields_mapping.items():
        bad_fields = [field for field in v if field in dataset['train'].features and field!=k]
        if bad_fields:
            dataset=dataset.rename_column(bad_fields[0], k)
    return dataset

def align_fields_MultipleChoice(dataset):
    fields_mapping={'inputs':['sentence1','question']}
    for k,v in fields_mapping.items():
        bad_fields = [field for field in v if field in dataset['train'].features and field!=k]
        if bad_fields:
            dataset=dataset.rename_column(bad_fields[0], k)
    return dataset

def process_labels(dataset):
    labels=pd.Series(dataset['train']['labels']).value_counts().reset_index()
    label_to_index=fc.flip(labels['index'].to_dict())
    def tokenize_labels(x):
        x['labels']=label_to_index.get(x['labels'],max(label_to_index.values())+1)
        return x
    dataset=dataset.map(tokenize_labels)
    return dataset

def get_name(dataset):
    return str(dataset.cache_files).split('/.cache/huggingface/datasets/')[-1].split('/')[0]

def dataset_deduplicate(dataset,subset=None):
    return DatasetDict({k:Dataset.from_pandas(
        pd.DataFrame(dataset[k]).drop_duplicates(subset=subset),preserve_index=False
    ) for k in dataset})

def task_type(x):
    if t0_task_type(x):
        return t0_task_type(x)

    if x.dataset_name in {'bigbench','blimp','hendrycks_test'}:
        return 'MultipleChoice'
    if x.dataset_name in {'relbert/lexical_relation_classification','lex_glue'}:
        return 'Classification'

    tc=x.task_categories
    if tc and len(tc)==1 and tc[0] in categories_mapping:
        return categories_mapping[tc[0]]
    if tc and set(tc)=={'multiple-choice','question-answering'}:
        return categories_mapping['multiple-choice']


def get_blimp():
    d=defaultdict(list)
    for c in get_dataset_config_names('blimp'):
        df=pd.DataFrame(load_dataset('blimp',c))['train']
        d[df.iloc[0]['linguistics_term']]+=[df]
    l=[]
    for c,l_df in d.items():
        ds=DatasetDict(train=Dataset.from_pandas(pd.concat(l_df)))
        setattr(ds,'task_config',c)
        l+=[ds]


@dataclass
class TaskParser:
    max_choices:int=None

    def normalize_anli(self, dataset):
        l=[]
        for i in '123':
            split=[f'train_r{i}',f'dev_r{i}',f'test_r{i}']
            ds=fc.project(dataset,split)
            ds=DatasetDict({k.split('_')[0]:v for k,v in ds.items()})
            setattr(ds,'task_config',f'split:{i}')
            l+=[align_splits(ds)]
        return l
    
    def normalize_conll2003(self, dataset):
        l=[]
        labels=['pos_tags', 'chunk_tags', 'ner_tags']
        for y in labels:
            ds=dataset.rename_column(y,'labels')
            setattr(ds,'task_config',f'label:{y}')
            l+=[ds.remove_columns([x for x in labels if x!=y])]
        return l

    def normalize_sick(self, dataset):
        l=[]
        labels=['entailment_AB','entailment_BA','label','relatedness_score']
        for y in labels:
            ds=dataset.rename_column(y,'labels')
            setattr(ds,'task_config',f'label:{y}')
            l+=[ds.remove_columns([x for x in labels if x!=y])]
        return l

    def normalize_blimp(self, dataset):

        def add_label(x):
            x['label']=0
            x['inputs']=''
            return x
        dataset=dataset.map(add_label).\
        rename_column('sentence_good','choice0').\
        rename_column('sentence_bad','choice1')
        return dataset
    
    def normalize_hendrycks_test(self, dataset):
        def reformat(x):
            for i in range(4):
                x[f'choice{i}']=x['choices'][i]
            del x['choices']
            return x  
        return dataset.map(reformat).rename_column('answer','labels')
        
    def normalize_bigbench(self, dataset):
        try:
            minimum_answer_counts=min(
                [ds.with_format("pandas")["multiple_choice_targets"].map(len).min() 
                 for ds in dataset.values()
                ]
            )
            assert minimum_answer_counts<11
            print('minimum_answer_counts:',minimum_answer_counts)
        except:
            raise ValueError('Unsupported bigbench format')

        def cap_options(x,n=None):
            if n!=None:
                n=n-1
            nz=np.array(np.nonzero(x['multiple_choice_scores'])).flatten()
            assert len(nz)==1
            nz=nz[0]
            l=x['multiple_choice_targets']
            y=x['multiple_choice_scores']
            x['multiple_choice_targets']=[l[nz]]+ (l[:nz] + l[(nz + 1):])[:n]
            x['multiple_choice_scores'] =[y[nz]]+ (y[:nz] + y[(nz + 1):])[:n]
            return x

        def reformat(x):
            if self.max_choices:
                n_options = min(minimum_answer_counts, self.max_choices)
            else:
                n_options = minimum_answer_counts
            x=cap_options(x,n_options)
            x['labels']=np.argmax(x['multiple_choice_scores'])
            for i,o in enumerate(x['multiple_choice_targets']):
                x[f'choice{i}']=o
            return x
        dataset= dataset.map(reformat)
        dataset=dataset_deduplicate(dataset,subset=['inputs','choice0'])
        return dataset

    def parse(self, dataset,dataset_name=None, task_type=None):
        if not dataset_name:
            dataset_name=get_name(dataset)
            print('name:',dataset_name)
        if hasattr(self, f'normalize_{dataset_name}'):
            dataset=getattr(self, f'normalize_{dataset_name}')(dataset)

        datasets = [dataset] if type(dataset)!=list else dataset
        l=[]
        for dataset in datasets:
            dataset=align_splits(dataset)
            dataset=fix_splits(dataset)
            dataset=align_fields(dataset)
            dataset=process_labels(dataset)
            if task_type=='MultipleChoice':
                dataset=align_fields_MultipleChoice(dataset)
            l+=[dataset]
        return l