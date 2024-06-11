has_test = True
deterministic = True
use_custom_worker_init = True
base_seed = 112358
log_interval = 20


__BATCHSIZE = 4
__NUM_EPOCHS = 30
__NUM_ITERS = 0
__EPOCH_BASED = True

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    num_epochs=__NUM_EPOCHS,
    num_iters=__NUM_ITERS,
    epoch_based=__EPOCH_BASED,
    grad_acc_step=1,
    lr=0.00003,
    sche_usebatch=False,
    optimizer=dict(
        mode="adamw",
        group_mode="finetune",
        cfg=dict(
            weight_decay=0.0005,
            diff_factor=0.1,
        ),
    ),
    scheduler=dict(
        warmup=dict(
            num_iters=0,
        ),
        mode="constant",
        cfg=dict(),
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    save_results=False,
)

datasets = dict(
    train=dict(shape=dict(h=384, w=384)),
    val=dict(shape=dict(h=384, w=384)),
    test=dict(shape=dict(h=384, w=384)),
)
