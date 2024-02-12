import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer
from utils import compare_dict, normalize


def load_hf_model(model_path, device="cpu", use_fp16=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    if use_fp16:
        model = model.half()

    return tokenizer, model


def hf_forward(model, tokenizer, input_strs, device="cpu"):
    input_ids = tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).to("cuda:1")
    output_hf = model(**input_ids, return_dict=True)
    hf_dense_output = output_hf.dense_output
    hf_dense_output = normalize(hf_dense_output)
    hf_colbert_output = output_hf.colbert_output
    hf_colbert_output = [normalize(vec) for vec in hf_colbert_output]
    hf_sparse_output = output_hf.sparse_output
    return hf_dense_output, hf_colbert_output, hf_sparse_output


def load_fg_model(model_path, device="cpu", use_fp16=False):
    model_flag = BGEM3FlagModel(model_path, device=device, use_fp16=use_fp16)
    return model_flag


def fg_forward(model, input_strs, device="cpu"):
    embeddings = model.encode(input_strs, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    flag_dense_output = embeddings["dense_vecs"]
    flag_colbert_output = embeddings["colbert_vecs"]
    flag_sparse_output = embeddings["lexical_weights"]
    return flag_dense_output, flag_colbert_output, flag_sparse_output


def main():
    # Set up test strings
    test_strings = ["I'm an example sentence.", "我是另一个测试句子。"]

    # Set up model configs
    hf_model_dir = "/workspace/nsr_cls/transformers_bge_m3/export_bge_m3"
    fg_model_dir = "/large-storage/model/bge-m3"

    device = "cuda:1"
    use_fp16 = True

    tokenizer, hf_model = load_hf_model(hf_model_dir, device=device, use_fp16=use_fp16)
    fg_model = load_fg_model(fg_model_dir, device=device, use_fp16=use_fp16)

    hf_dense_output, hf_colbert_output, hf_sparse_output = hf_forward(hf_model, tokenizer, test_strings, device=device)
    fg_dense_output, fg_colbert_output, fg_sparse_output = fg_forward(fg_model, test_strings, device=device)

    # Convert FG outputs to tensors
    fg_dense_output = torch.tensor(fg_dense_output, dtype=hf_dense_output.dtype, device=hf_dense_output.device)
    fg_colbert_output = [
        torch.tensor(vec, dtype=hf_colbert_output[0].dtype, device=hf_colbert_output[0].device)
        for vec in fg_colbert_output
    ]

    # compare outputs
    print("=== Comparing outputs ===")
    print("dense_output:", torch.allclose(hf_dense_output, fg_dense_output))
    print(
        "colbert_output:",
        [torch.allclose(hf_colbert_output[i], fg_colbert_output[i]) for i in range(len(hf_colbert_output))],
    )
    print(
        "sparse_output:",
        [compare_dict(hf_sparse_output[i], fg_sparse_output[i]) for i in range(len(hf_sparse_output))],
    )


if __name__ == "__main__":
    main()
