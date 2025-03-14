class Params:

    # Data Transformation Stage
    normalize_mean = (0.5,)
    normalize_std = (0.5,)

    # Model Ingestion Stage
    beta_start = 1e-4
    beta_end = 0.02
    n_timesteps = 1000
    img_size = 64
    device = "cuda"
    channel_size = 3
    label_size = 10
    embed_size = 256

    # Training Stage
    learning_rate = 3e-4
    batch_size = 100
    noise_dim = 100
    betas = (0.5, 0.99)
    epochs = 4



