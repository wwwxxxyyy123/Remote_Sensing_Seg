SEGFORMER_CONFIGS = {
    'b0': {
        'embed_dims': [32, 64, 160, 256],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [2, 2, 2, 2],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    },
    'b1': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [2, 2, 2, 2],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    },
    'b2': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [3, 4, 6, 3],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    },
    'b3': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [3, 4, 18, 3],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    },
    'b4': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [3, 8, 27, 3],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    },
    'b5': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'num_layers': [3, 6, 40, 3],
        'reduction_ratios': [8, 4, 2, 1],
        'mlp_ratios': [4, 4, 4, 4],
    }
}