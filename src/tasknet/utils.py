from datasets import DatasetDict


def train_validation_test_split(dataset, train_ratio=0.8, val_test_ratio=0.5):
    train_testvalid = dataset.train_test_split(test_size=1 - train_ratio)
    test_valid = train_testvalid["test"].train_test_split(test_size=val_test_ratio)
    dataset = DatasetDict(
        train=train_testvalid["train"],
        validation=test_valid["test"],
        test=test_valid["train"],
    )
    return dataset
