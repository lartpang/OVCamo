has_test = True
deterministic = True
use_custom_worker_init = True
base_seed = 112358
log_interval = 20


__BATCHSIZE = 2
__NUM_EPOCHS = 50
__NUM_ITERS = 0
__EPOCH_BASED = True

train = dict(
    save_interval=-1,
    batch_size=__BATCHSIZE,
    num_workers=2,
    num_epochs=__NUM_EPOCHS,
    num_iters=__NUM_ITERS,
    epoch_based=__EPOCH_BASED,
    use_amp=True,
    grad_acc_step=1,
    lr=0.00002,
    sche_usebatch=False,
    optimizer=dict(
        mode="adamw",
        group_mode="finetune",
        set_to_none=True,
        cfg=dict(
            weight_decay=0.0001,
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
