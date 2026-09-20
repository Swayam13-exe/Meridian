#!/usr/bin/env bash
# training/run_gigpo_smoke_test.sh
# ==================================
# Identical to run_grpo_smoke_test.sh in every setting EXCEPT the ones
# that are genuinely about the algorithm difference: algorithm.adv_estimator
# and the three gigpo-specific keys below. That's deliberate, not an
# oversight -- diff this file against run_grpo_smoke_test.sh and the only
# differences should be those four lines plus the experiment name. If any
# other line differs between the two scripts, the eventual GRPO-vs-GiGPO
# comparison stops being clean, since any difference we observe could be
# explained by something other than the algorithm itself.
#
# Run from the Meridian repo root:
#   bash training/run_gigpo_smoke_test.sh

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

REPO_ROOT=$(pwd)
DATA_DIR="$HOME/data/meridian"

train_data_size=4
val_data_size=4
group_size=2
gigpo_mode="mean_std_norm"  # or "mean_norm" -- see the GiGPO paper's ablation for the tradeoff

python3 training/prepare_meridian_data.py \
    --train-size $train_data_size \
    --val-size $val_data_size \
    --out-dir $DATA_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$gigpo_mode \
    env.env_name=meridian/WorkforceEnv \
    env.seed=0 \
    env.max_steps=40 \
    env.history_length=5 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    +env.meridian.task_pool=[easy,medium,hard,crisis] \
    +ray_init.runtime_env.working_dir=$REPO_ROOT \
    +ray_init.runtime_env.worker_process_setup_hook=training.patch_env_registry.apply_patch \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='meridian' \
    trainer.experiment_name='gigpo_smoke_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True $@