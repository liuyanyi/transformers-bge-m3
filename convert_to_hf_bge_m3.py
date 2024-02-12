import argparse
import logging
import os
from collections import OrderedDict

import torch
from hf_bge_m3.configuration_bge_m3 import BgeM3Config
from hf_bge_m3.modeling_bge_m3 import BgeM3Model
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to Original BGE-M3 Model")
    parser.add_argument("output_dir", type=str, help="Path to Output Directory")
    parser.add_argument("--export_st", action="store_true", help="Export Sentence Transformer Model")
    return parser.parse_args()


def main():
    args = parse_args()

    # Register the BGE-M3 model
    BgeM3Config.register_for_auto_class()
    BgeM3Model.register_for_auto_class("AutoModel")

    # Load the BGE-M3 model
    logger.info(f"Loading BGE-M3 model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    bge_m3_config = BgeM3Config.from_pretrained(args.model_dir)

    unused_tokens = [
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]

    bge_m3_config.unused_tokens = unused_tokens

    logger.info("Below warning about different model_type and weight not initialized is expected")
    model = BgeM3Model.from_pretrained(args.model_dir, config=bge_m3_config)

    # Check for the linear layers
    colbert_path = os.path.join(args.model_dir, "colbert_linear.pt")
    sparse_path = os.path.join(args.model_dir, "sparse_linear.pt")

    if not os.path.exists(colbert_path) or not os.path.exists(sparse_path):
        raise ValueError("Linear weights not found in the model directory")

    colbert_state_dict = torch.load(colbert_path, map_location="cpu")
    sparse_state_dict = torch.load(sparse_path, map_location="cpu")

    # set the linear weights
    model.colbert_linear.load_state_dict(colbert_state_dict)
    model.sparse_linear.load_state_dict(sparse_state_dict)

    # Check the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the model
    logger.info(f"Saving hf model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    del model, tokenizer

    if not args.export_st:
        return

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize, Pooling, Transformer

    # Process sentence transformer model
    modules = OrderedDict()
    transformer = Transformer(
        model_name_or_path=args.output_dir,
        model_args={
            "trust_remote_code": True,
        },
        max_seq_length=8192,
        do_lower_case=False,
    )

    pooling = Pooling(
        word_embedding_dimension=1024,
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_sqrt_len_tokens=False,
    )

    norm = Normalize()

    modules["0"] = transformer
    modules["1"] = pooling
    modules["2"] = norm

    sentence_transformer_model = SentenceTransformer(modules=modules)

    sentence_transformer_model.save(args.output_dir, create_model_card=False)

    logger.info("Model saved successfully")


if __name__ == "__main__":
    main()
