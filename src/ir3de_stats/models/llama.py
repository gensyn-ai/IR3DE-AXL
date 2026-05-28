import torch
from ir3de_stats.models.modeling_llama import LlamaConfig, LlamaForCausalLM


class LlamaWrapper(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "eager",
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
    ):
        super().__init__()
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=4 * hidden_size,
            vocab_size=vocab_size,
        )
        config.torch_dtype = torch.bfloat16
        config._attn_implementation = attn_implementation
        self._model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)  # type: ignore

    def forward(self, input_ids: torch.Tensor, expert_id=None, layer_id=None):
        outputs = self._model(input_ids=input_ids, expert_id=expert_id, layer_id=layer_id)
        if isinstance(outputs, tuple):
            all_p = outputs[1]
            all_u = outputs[2]
            outputs = outputs[0]
            return outputs.logits, all_p, all_u
        return outputs.logits

    def generate(self, input_ids, expert_id=None, **kwargs):
        if expert_id is not None:
            original_forward = self._model.forward  # Store original forward method
            
            # Create a wrapper that injects expert_id
            def forward_with_expert_id(*args, **forward_kwargs):
                forward_kwargs['expert_id'] = expert_id
                return original_forward(*args, **forward_kwargs)
            
            self._model.forward = forward_with_expert_id  # Temporarily replace the forward method
            
            try:
                result = self._model.generate(input_ids=input_ids, **kwargs)  # Call generate with modified forward
            finally:
                self._model.forward = original_forward  # Restore original forward method
                
            return result
        else:
            return self._model.generate(input_ids=input_ids, **kwargs)


# additional parameter of intermediate_size
# That allows keeping the same hidden_size and manipulating the intermediate_size for different model sizes
class LlamaWrapperWithMLPSize(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "eager",
        config=None,
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
    ):
        super().__init__()
        if config is None:
            config = LlamaConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
            )
            config.torch_dtype = torch.bfloat16
            config._attn_implementation = attn_implementation
        
        self._model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)  # type: ignore

    def forward(self, input_ids: torch.Tensor, expert_id=None, layer_id=None):
        outputs = self._model(input_ids=input_ids, expert_id=expert_id, layer_id=layer_id)
        if isinstance(outputs, tuple):
            all_p = outputs[1]
            all_u = outputs[2]
            outputs = outputs[0]
            return outputs.logits, all_p, all_u
        return outputs.logits

    def generate(self, input_ids, expert_id=None, **kwargs):
        if expert_id is not None:
            original_forward = self._model.forward  # Store original forward method
            
            # Create a wrapper that injects expert_id
            def forward_with_expert_id(*args, **forward_kwargs):
                forward_kwargs['expert_id'] = expert_id
                return original_forward(*args, **forward_kwargs)
            
            self._model.forward = forward_with_expert_id  # Temporarily replace the forward method
            
            result = self._model.generate(input_ids=input_ids, **kwargs)  # Call generate with modified forward
            self._model.forward = original_forward  # Restore original forward method
                
            return result
        else:
            return self._model.generate(input_ids=input_ids, **kwargs)
    