def emformer_rnnt(
    segment_length: int = 64,
    right_context_length: int = 0,
    left_context_length: int = 30,
    num_layers: int = 20,
):
    """Emformer RNN-T from the torchaudio examples/avsr recipe.

    segment_length/right_context_length set the streaming cadence:
    Emformer.infer consumes exactly segment_length + right_context_length
    fused frames (25 fps) per call. Weight shapes do not depend on these
    values, so one checkpoint loads into any cadence.
    """
    try:
        from torchaudio.models.rnnt import emformer_rnnt_model
    except ImportError:
        from torchaudio.models import emformer_rnnt_model

    return emformer_rnnt_model(
        input_dim=512,
        encoding_dim=1024,
        num_symbols=1024,
        segment_length=segment_length,
        right_context_length=right_context_length,
        time_reduction_input_dim=128,
        time_reduction_stride=1,
        transformer_num_heads=4,
        transformer_ffn_dim=2048,
        transformer_num_layers=num_layers,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        transformer_left_context_length=left_context_length,
        transformer_max_memory_size=0,
        transformer_weight_init_scale_strategy="depthwise",
        transformer_tanh_on_mem=True,
        symbol_embedding_dim=512,
        num_lstm_layers=3,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
    )


def emformer_rnnt_device(
    segment_length: int = 32,
    right_context_length: int = 4,
    left_context_length: int = 30,
    num_layers: int = 12,
):
    """The published device_avsr (blog "Small") configuration, derived from
    the tutorial JIT archive: 12 Emformer layers at dim 256, ffn 1024,
    4 heads, segment 32 + right context 4 (~1.4 s window at 25 fps)."""
    try:
        from torchaudio.models.rnnt import emformer_rnnt_model
    except ImportError:
        from torchaudio.models import emformer_rnnt_model

    return emformer_rnnt_model(
        input_dim=512,
        encoding_dim=1024,
        num_symbols=1024,
        segment_length=segment_length,
        right_context_length=right_context_length,
        time_reduction_input_dim=256,
        time_reduction_stride=1,
        transformer_num_heads=4,
        transformer_ffn_dim=1024,
        transformer_num_layers=num_layers,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        transformer_left_context_length=left_context_length,
        transformer_max_memory_size=0,
        transformer_weight_init_scale_strategy="depthwise",
        transformer_tanh_on_mem=True,
        symbol_embedding_dim=512,
        num_lstm_layers=3,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
    )
