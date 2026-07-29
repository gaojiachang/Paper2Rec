# Paper2Rec

精简版推荐模型代码库，仅保留强化学习目标与独立推荐模型实现。

## 目录

- `rl/`：PPO、DPO、GRPO。
- `rec/`：DIN、SIM、SASRec、HyFormer、SlimPer。

每个推荐模型均为独立的 PyTorch 文件，不包含数据集、样本、训练流水线或实验输出。

## 环境

```bash
uv venv .venv
uv pip install torch
```

各模型文件可直接运行其内置的最小示例，例如：

```bash
python rec/DIN.py
python rec/SIM.py
python rec/SASRec.py
```
