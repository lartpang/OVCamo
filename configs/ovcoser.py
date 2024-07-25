has_test = True
deterministic = True
use_custom_worker_init = True
base_seed = 112358
log_interval = 20


__BATCHSIZE = 4
__NUM_EPOCHS = 30

train = dict(
    batch_size=__BATCHSIZE,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=0,
    grad_acc_step=1,
    num_workers=2,
    lr=0.00003,
    sche_usebatch=False,
    optimizer=dict(
        mode="adamw",
        group_mode="finetune",
        set_to_none=True,
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
    input_hw=[384, 384],
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    save_results=True,
    input_hw=[384, 384],
)
