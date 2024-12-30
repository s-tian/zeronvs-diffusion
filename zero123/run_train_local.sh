python main.py \
    -t \
    --base configs/finetune_mimicgen.yaml \
    --gpus "0,1,2,3" \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --finetune_from zeronvs.ckpt \
    --enable_look_for_checkpoints False \
    data.params.train_config.batch_size=64 \
    data.params.val_config.rate=1 \
    lightning.trainer.val_check_interval=750\
    model.params.conditioning_config.params.mode='7dof_quantile_scale' \
    model.params.conditioning_config.params.embedding_dim=19 \
    lightning.trainer.accumulate_grad_batches=2 \
    lightning.callbacks.image_logger.params.log_first_step=False \
    lightning.modelcheckpoint.params.every_n_train_steps=500 \
    lightning.callbacks.image_logger.params.batch_frequency=100