

import os


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class HyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = _env_bool('HP_EXTRA', False)
    data_dir = os.getenv('HP_DATA_DIR', 'datasets')
    extra_dir = os.getenv('HP_EXTRA_DIR', 'additional')
    ckpt_dir = os.getenv('HP_CKPT_DIR', 'ckpt')

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = _env_int('HP_BATCH_SIZE', 2) # actually batchsize * Seqlen
    seq_len = _env_int('HP_SEQ_LEN', 32)

    test_batch = _env_int('HP_TEST_BATCH', 1)
    n_sample = _env_int('HP_N_SAMPLE', 64)

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = _env_int('HP_LOG_INTERVAL', 5000)
    save_interval = _env_int('HP_SAVE_INTERVAL', 10000)

    use_binary_feature = _env_bool('HP_USE_BINARY_FEATURE', False)
    score_cut = _env_int('HP_SCORE_CUT', 300) # to save
    save_start_score = _env_int('HP_SAVE_START_SCORE', 100)

    # Rollout
    max_ep = _env_int('HP_MAX_EP', 1000)
    n_rollout = _env_int('HP_N_ROLLOUT', 200)
    seed = _env_int('HP_SEED', 0)

    n_workers = _env_int('HP_N_WORKERS', 0)

class RNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'

    extra = _env_bool('RNN_EXTRA', False)
    data_dir = os.getenv('RNN_DATA_DIR', 'datasets')
    extra_dir = os.getenv('RNN_EXTRA_DIR', 'additional')
    ckpt_dir = os.getenv('RNN_CKPT_DIR', 'ckpt')

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = _env_int('RNN_BATCH_SIZE', 1) # actually batchsize * Seqlen
    test_batch = _env_int('RNN_TEST_BATCH', 1)
    seq_len = _env_int('RNN_SEQ_LEN', 32)
    n_sample = _env_int('RNN_N_SAMPLE', 64)

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size
    rnn_hunits = 256
    log_interval = _env_int('RNN_LOG_INTERVAL', 1000)
    save_interval = _env_int('RNN_SAVE_INTERVAL', 2000)

    max_step = _env_int('RNN_MAX_STEP', 100000)
    seed = _env_int('RNN_SEED', 0)

    n_workers = _env_int('RNN_N_WORKERS', 0)

class VAEHyperParams:
    vision = 'VAE'

    extra = _env_bool('VAE_EXTRA', False)
    data_dir = os.getenv('VAE_DATA_DIR', 'datasets')
    extra_dir = os.getenv('VAE_EXTRA_DIR', 'additional')
    ckpt_dir = os.getenv('VAE_CKPT_DIR', 'ckpt')

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = _env_int('VAE_BATCH_SIZE', 64)
    test_batch = _env_int('VAE_TEST_BATCH', 12)
    n_sample = _env_int('VAE_N_SAMPLE', 64)

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size

    log_interval = _env_int('VAE_LOG_INTERVAL', 5000)
    save_interval = _env_int('VAE_SAVE_INTERVAL', 10000)

    max_step = _env_int('VAE_MAX_STEP', 2000000)

    n_workers = _env_int('VAE_N_WORKERS', 0)
