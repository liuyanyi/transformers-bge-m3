# BGE-M3 in HuggingFace Transformer

> [!Warning]
> This is not an official implementation of BGE-M3. Official implementation can be found in [Flag Embedding](https://github.com/FlagOpen/FlagEmbedding) project.

## Introduction

BGE-M3 is a versatile embedding model that can perform dense retrieval, multi-vector retrieval, and sparse retrieval. It is also multilingual and supports multiple granularities. This model support three functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval.

Currently, dense embedding is supported in HuggingFace Transformer. Multi-vector retrieval and sparse retrieval can only use in [Flag Embedding](https://github.com/FlagOpen/FlagEmbedding) project.

In this repository, i use a custom model to integrate full functionality of BGE-M3 in HuggingFace Transformer.

## Convert Model

1. A converted model has been upload to [HuggingFace](https://huggingface.co/liuyanyi/bge-m3-hf)

2. If you have an offical weight, you can convert it ot our implementation

```bash
python convert_to_hf_bge_m3.py <path_to_bge_m3_model> <output_dir> --export_st
```

this will convert the BGE-M3 model to our format and save it to the output directory. The `--export_st` flag is used to export the sentence transformer model.

## Use BGE-M3 in HuggingFace Transformer

```python
from transformers import AutoModel, AutoTokenizer

# Trust remote code is required to load the model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

input_str = "Hello, world!"
input_ids = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True)

output = model(**input_ids, return_dict=True)

dense_output = output.dense_output # To align with Flag Embedding project, a normalization is required
colbert_output = output.colbert_output # To align with Flag Embedding project, a normalization is required
sparse_output = output.sparse_output
```

## Alignment with Flag Embedding Project

Refer to the original implementation, this repository's colbert and sparse implementation has made some minor modifications. So we need to ensure that our model is consistent with the model in the Flag Embedding project. Here is a small script `test_with_flag_embedding.py`

First modify the `test_with_flag_embedding.py` file to load your model and test data.

```bash
python test_with_flag_embedding.py
```

If everything is correct, the output should like this:

```text
=== Comparing outputs ===
dense_output: True
colbert_output: [True, True]
sparse_output: [True, True]
```

## References

- [Official BGE-M3 Weight](https://huggingface.co/BAAI/bge-m3)
- [Flag Embedding](https://github.com/FlagOpen/FlagEmbedding)
- [HuggingFace Transformer](https://github.com/huggingface/transformers)