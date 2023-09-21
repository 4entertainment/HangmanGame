```
import datasets
from datasets import load_dataset


dataset = load_dataset("lhoestq/demo1")
datasets.utils.HF_HOME = "/path/to/huggingface/cache"



def main():
    # Specify the dataset name you want to download from Hugging Face.
    dataset_name = "lhoestq/demo1"  # Change this to your desired dataset.

    # Load the dataset from Hugging Face.
    dataset = load_dataset(dataset_name)

    # Print some basic information about the dataset.
    print(f"Dataset name: {dataset_name}")
    print(f"Number of splits: {len(dataset)}")
    print(f"Available splits: {dataset.keys()}")

    # View the first few examples from the dataset.
    print("\nSample data:")
    for i, example in enumerate(dataset["train"][:5]):
        print(f"Example {i + 1}:")
        print(f"Question: {example['question']}")
        print(f"Context: {example['context']}\n")
        print(f"Answer: {example['answers']['text'][0]}\n")


if __name__ == "__main__":
    main()

```


(sampleproject) berkin@berkin:~/Desktop/sampleproject$ python main.py
Traceback (most recent call last):
  File "/home/berkin/Desktop/sampleproject/main.py", line 5, in <module>
    dataset = load_dataset("lhoestq/demo1")
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/berkin/anaconda3/envs/sampleproject/lib/python3.11/site-packages/datasets/load.py", line 2129, in load_dataset
    builder_instance = load_dataset_builder(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/home/berkin/anaconda3/envs/sampleproject/lib/python3.11/site-packages/datasets/load.py", line 1815, in load_dataset_builder
    dataset_module = dataset_module_factory(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/berkin/anaconda3/envs/sampleproject/lib/python3.11/site-packages/datasets/load.py", line 1512, in dataset_module_factory
    raise e1 from None
  File "/home/berkin/anaconda3/envs/sampleproject/lib/python3.11/site-packages/datasets/load.py", line 1468, in dataset_module_factory
    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({type(e).__name__})")
ConnectionError: Couldn't reach 'lhoestq/demo1' on the Hub (SSLError)
