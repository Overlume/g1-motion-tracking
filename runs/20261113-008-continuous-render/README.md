# 20261113-008 continuous-render (fixed replay videos)

目的：修复"录像约 3 秒后动作反复重播"的问题，产出能覆盖完整 rollout、反映真实行为的连续录像。

做法：1 env、逐帧检查 dones 并记录、--ignore-terminations（不做自动 reset），录满 999 帧（19.98 s），960x540、50 fps。

命令（项目根目录）：

    for spec in "data/processed/fight1_s2_134s_20s.npz fight_s2_cont" "data/processed/fight1_s3_000s_20s.npz fight_s3_cont"; do
      set -- $spec
      MUJOCO_GL=egl environment/.venv/bin/python scripts/replay_policy.py \
        --checkpoint-file scratch/model_49999.pt --motion-file $1 \
        --num-envs 1 --video-length 999 --width 960 --height 540 \
        --ignore-terminations --tag $2
    done

产物：

- outputs/videos/model_49999/960x540/fight_s2_cont/rl-video-step-0.mp4（3,964,074 B）+ episode_log.json（reset_count 0）+ contact_2fps.png
- outputs/videos/model_49999/960x540/fight_s3_cont/rl-video-step-0.mp4（3,513,100 B）+ episode_log.json（reset_count 0）+ contact_2fps.png
- 日志 environment/logs/render_cont.log（S2 用时 122 s，S3 用时 103 s）。

看到的内容：S2 在约 4.3 s 摔倒，之后在地上挣扎到 20 s；S3 前 14 s 能打，13.92 s 摔倒。参考动作（绿色 ghost）全程不受影响地继续演。

限制：单 env 只是定性强证据，量化结论以 004/005/007/009 的 summary.json 为准。