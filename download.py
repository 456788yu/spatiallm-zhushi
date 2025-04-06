from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("manycore-research/SpatialLM-Testset")
ds.save_to_disk('data')