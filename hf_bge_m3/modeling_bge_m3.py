from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from transformers.models.xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)

from .configuration_bge_m3 import BgeM3Config


@dataclass
class BgeM3ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    dense_output: torch.FloatTensor = None
    colbert_output: Optional[List[torch.FloatTensor]] = None
    sparse_output: Optional[Dict[int, float]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BgeM3Model(XLMRobertaPreTrainedModel):
    config_class = BgeM3Config

    def __init__(self, config: BgeM3Config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # TODO: Check the dtype of these linear layers
        self.colbert_linear = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size if config.colbert_dim is None else config.colbert_dim,
        )
        self.sparse_linear = nn.Linear(in_features=config.hidden_size, out_features=1)
        self.sentence_pooling_method = config.sentence_pooling_method

        self.init_weights()

    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = False):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.config.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device,
        )
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = self.config.unused_tokens
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    def colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def _process_token_weights(self, token_weights, input_ids, mask):
        token_weights = token_weights.squeeze(-1)
        # conver to dict
        all_result = []
        unused_tokens = self.config.unused_tokens
        unused_tokens = torch.tensor(unused_tokens, device=input_ids.device)

        # 获取有效的 token 的索引
        valid_indices = ~torch.isin(input_ids, unused_tokens)
        # weight必须大于0
        valid_indices = (valid_indices & (token_weights > 0)).bool()
        # 结合 attention mask，获取有效的 token 的索引
        valid_indices = (valid_indices & mask).bool()

        for i, valid in enumerate(valid_indices):
            result = defaultdict(int)

            # 获取有效的 weights 和 ids
            valid_weights = token_weights[i][valid]
            valid_ids = input_ids[i][valid]

            # 获取每个 id 的最大权重
            unique_ids, inverse_indices = torch.unique(valid_ids, return_inverse=True)

            # 使用一个循环来找到每个 unique id 的最大权重
            for i in range(unique_ids.shape[0]):
                id_mask = inverse_indices == i
                result[str(unique_ids[i].item())] = valid_weights[id_mask].max().item()

            all_result.append(result)
        # token_weights = np.ceil(token_weights * 100)
        # for w, idx, num in zip(token_weights, input_ids, tokens_num):
        #     r = defaultdict(int)
        #     token_weight = w[:num]
        #     idx = idx[:num]

        #     for t_w, t_idx in zip(token_weight, idx):
        #         if t_idx.item() not in unused_tokens:
        #             t_idx = str(t_idx.item())
        #             if t_w > r[t_idx]:
        #                 r[t_idx] = t_w.item()

        #     result.append(r)

        # if idx not in unused_tokens and w > 0:
        #     idx = str(idx)
        #     # w = int(w)
        #     if w > result[idx]:
        #         result[idx] = w
        return all_result

    def _process_colbert_vecs(self, colbert_vecs, tokens_num) -> List[torch.Tensor]:
        # delte the vectors of padding tokens
        vecs = []
        for i in range(len(tokens_num)):
            vecs.append(colbert_vecs[i, : tokens_num[i] - 1])
        return vecs

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BgeM3ModelOutput]:
        roberta_output: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = roberta_output.last_hidden_state
        dense_output = self.dense_embedding(last_hidden_state, attention_mask)

        tokens_num = attention_mask.sum(dim=1)
        colbert_output = self.colbert_embedding(last_hidden_state, attention_mask)
        colbert_output = self._process_colbert_vecs(colbert_output, tokens_num)

        sparse_output = self.sparse_embedding(last_hidden_state, input_ids)
        sparse_output = self._process_token_weights(sparse_output, input_ids, attention_mask)

        if not return_dict:
            return (
                last_hidden_state,
                roberta_output.pooler_output,
                dense_output,
                colbert_output,
                sparse_output,
                roberta_output.hidden_states,
                roberta_output.past_key_values,
                roberta_output.attentions,
                roberta_output.cross_attentions,
            )

        return BgeM3ModelOutput(
            last_hidden_state=last_hidden_state,
            dense_output=dense_output,
            pooler_output=roberta_output.pooler_output,
            colbert_output=colbert_output,
            sparse_output=sparse_output,
            hidden_states=roberta_output.hidden_states,
            past_key_values=roberta_output.past_key_values,
            attentions=roberta_output.attentions,
            cross_attentions=roberta_output.cross_attentions,
        )
