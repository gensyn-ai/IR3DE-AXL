from ir3de_stats.models.llama import LlamaWrapper, LlamaWrapperWithMLPSize

def get_llama_expert(num_parameters, config=None, attn_implementation="eager"):
    
    if num_parameters == 5e6:
        batch_size = 2.62e5
        max_lr = 5e-3
        return LlamaWrapper(
            hidden_size=272,
            num_attention_heads=8,
            num_hidden_layers=4,
        ), batch_size, max_lr
    
    if num_parameters == 7.5e6:
        batch_size = 2.62e5
        max_lr = 5e-3
        return LlamaWrapper(
            hidden_size=272,
            num_attention_heads=8,
            num_hidden_layers=6
        ), batch_size, max_lr
    
    if num_parameters == 1e7:
        batch_size = 2.62e5
        max_lr = 5e-3
        return LlamaWrapper(
            hidden_size=320,
            num_attention_heads=10,
            num_hidden_layers=6,
        ), batch_size, max_lr
    
    if num_parameters == 1.25e7:
        batch_size = 2.62e5
        max_lr = 5e-3
        return LlamaWrapper(
            hidden_size=330,
            num_attention_heads=11,
            num_hidden_layers=7,
        ), batch_size, max_lr
    
    if num_parameters == 1.5e7:
        batch_size = 2.62e5
        max_lr = 5e-3
        return LlamaWrapper(
            hidden_size=340,
            num_attention_heads=10,
            num_hidden_layers=8,
        ), batch_size, max_lr
    
    if num_parameters == 9e7:
        batch_size = 6.88e5
        max_lr = 6e-4
        return LlamaWrapperWithMLPSize(
            hidden_size=768,
            intermediate_size=2304,
            num_attention_heads=12,
            num_hidden_layers=12,
        ), batch_size, max_lr
    
    if num_parameters == 1.15e8:
        batch_size = 6.88e5
        max_lr = 6e-4
        return LlamaWrapper(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
        ), batch_size, max_lr
    
    if num_parameters == 1.35e8:
        batch_size = 6.88e5
        max_lr = 6e-4
        return LlamaWrapperWithMLPSize(
            hidden_size=768,
            intermediate_size=3840,
            num_attention_heads=12,
            num_hidden_layers=12,
        ), batch_size, max_lr

    if num_parameters == 1e9:
        batch_size = 1.72e6
        max_lr = 4e-4
        return LlamaWrapperWithMLPSize(
            config=config,
            hidden_size=-1,
            intermediate_size=-1,
            num_attention_heads=-1,
            num_hidden_layers=-1,
            attn_implementation=attn_implementation,
        ), batch_size, max_lr

    if num_parameters == 3e9:
        batch_size = 1.72e6
        max_lr = 3e-4
        return LlamaWrapperWithMLPSize(
            config=config,
            hidden_size=-1,
            intermediate_size=-1,
            num_attention_heads=-1,
            num_hidden_layers=-1,
            attn_implementation=attn_implementation,
        ), batch_size, max_lr
    
    raise ValueError(f"Unsupported number of parameters: {num_parameters}")
