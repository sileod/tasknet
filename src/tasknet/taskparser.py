from dataclasses import dataclass

split_mapping={
'train':['train_split','training'],
'validation':['validation','dev','eval_split']
}

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

fields_mapping={
    'sentence1':['premise','sentence','sentence1','text','head','question1','question'],
    'sentence2':['hypothesis','sentence2','tail','question2'],
    'labels':['label','labels','relation','gold_label']
    
}

def align_fields(x):
    for k,v in fields_mapping.items():
        bad_fields = [field for field in v if field in x and field!=k]
        if bad_fields:
            x[k]=x[bad_fields[0]]
            del x[bad_fields[0]]
    return x

def get_name(dataset):
    return str(dataset.cache_files).split('/.cache/huggingface/datasets/')[-1].split('/')[0]

def dataset_deduplicate(dataset,subset=None):
    return DatasetDict({k:Dataset.from_pandas(
        pd.DataFrame(dataset[k]).drop_duplicates(subset=subset),preserve_index=False
    ) for k in dataset})

def task_type(x):
    if x.config in {'boolq','cb'}:
        return 'Classification'
    if x.config in {'copa'}:
        return 'MultipleChoice'
    if x.dataset_name in {'bigbench','blimp','hendrycks_test'}:
        return 'MultipleChoice'
    if x.dataset_name in {'glue','anli','tweet_eval','pragmeval','relbert/lexical_relation_classification','metaeval/linguisticprobing'}:
        return 'Classification'
    if x.dataset_name in {'conll2003'}:
        return 'TokenClassification'

@dataclass
class TaskParser:
    #todo sick
    def normalize_anli(dataset):
        l=[]
        for i in '123':
            split=[f'train_r{i}',f'dev_r{i}',f'test_r{i}']
            ds=fc.project(dataset,split)
            ds=DatasetDict({k.split('_')[0]:v for k,v in ds.items()})
            setattr(ds,'task_config',f'split:{i}')
            l+=[align_splits(ds)]
        return l
    
    def normalize_conll2003(dataset):
        l=[]
        for y in ['pos_tags', 'chunk_tags', 'ner_tags']:
            ds=dataset.rename_column('pos_tags','labels')
            setattr(ds,'task_config',f'label:{y}')
            l+=[ds]
        return l
    
    def normalize_blimp(dataset):
        def add_label(x):
            x['label']=0
            x['s1']=''
            return x
        dataset=dataset.map(add_label).\
        rename_column('sentence_good','choice0').\
        rename_column('sentence_bad','choice1')
        return dataset
    
    def normalize_hendrycks_test(dataset):
        def reformat(x):
            for i in range(4):
                x[f'choice{i}']=x['choices'][i]
            del x['choices']
            return x  
        return dataset.map(reformat).rename_column('answer','labels')
        
    def normalize_bigbench(dataset):

        try:
            minimum_answer_counts=min(
                [ds.with_format("pandas")["multiple_choice_targets"].map(len).min() 
                 for ds in dataset.values()
                ]
            )
            assert minimum_answer_counts
            print('minimum_answer_counts:',minimum_answer_counts)
        except:
            raise ValueError('Unsupported bigbench format')
            
        def cap_options(x,n=None):
            nz=np.array(np.nonzero(x['multiple_choice_scores'])).flatten()
            assert len(nz)==1
            nz=nz[0]
            l=x['multiple_choice_targets']
            y=x['multiple_choice_scores']
            x['multiple_choice_targets']=[l[nz]]+ (l[:nz] + l[(nz + 1):])[:n]
            x['multiple_choice_scores'] =[y[nz]]+ (y[:nz] + y[(nz + 1):])[:n]
            return x
        
        def reformat(x):
            x=cap_options(x,minimum_answer_counts-1)
            x['labels']=np.argmax(x['multiple_choice_scores'])
            for i,o in enumerate(x['multiple_choice_targets']):
                x[f'choice{i}']=o
            return x
        dataset= dataset.map(reformat)
        dataset=dataset_deduplicate(dataset,subset=['inputs','choice0'])
        return dataset
    def parse(dataset,dataset_name=None):
        if not dataset_name:
            dataset_name=get_name(dataset)
            print('name:',dataset_name)
        if hasattr(TaskParser, f'normalize_{dataset_name}'):
            dataset=getattr(TaskParser, f'normalize_{dataset_name}')(dataset)
        if type(dataset)==list:
            dataset=dataset[0]
        dataset=align_splits(dataset)
        dataset=fix_splits(dataset)
        dataset=dataset.map(align_fields)
        return dataset