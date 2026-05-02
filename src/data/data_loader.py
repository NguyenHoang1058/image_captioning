from datasets import load_dataset, load_from_disk

def load_data_online():
    dataset = load_dataset("nlphuji/flickr30k")
    return dataset

def prepare_and_save(data_dir="data/processed"):
    dataset=load_dataset("nlphuji/flickr30k")["test"]

    split_1=dataset.train_test_split(test_size=0.2, seed=42)
    train=split_1["train"]
    temp=split_1["test"]

    split_2=temp.train_test_split(test_size=0.5, seed=42)
    val=split_2["train"]
    test=split_2["test"]

    train.save_to_disk(f"{data_dir}/train")
    val.save_to_disk(f"{data_dir}/val")
    test.save_to_disk(f"{data_dir}/test")
    
def load_data(data_dir="data/processed"):
    train=load_from_disk(f"{data_dir}/train")
    val=load_from_disk(f"{data_dir}/val")
    test=load_from_disk(f"{data_dir}/test")

    return train, val, test