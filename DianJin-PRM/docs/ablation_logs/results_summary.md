# 消融重训结果汇总
首次启动: Fri Oct  9 01:46:48 CST 2026

> ⚠️ **作废提示（2026-10-13）**：本文件中的 DPO β / epoch 消融结果（`beta_*`、`de2`、`de3`）为**旧「纯轨迹级」口径**，已作废。DPO 侧已按「加权融合」排序重做，最终数值见 `docs/DPO重做执行记录.md`（β margin 0.94/3.78/5.73/11.21，epoch margin 5.73/7.71/8.44）与 `docs/消融实验结果.md`。PRM 侧（`ep3`、`d500`、`d1000`）结果仍有效。

## ep3 (epochs=3, samples=全部)
test results (total=200, T=1.0):
  步骤准确率: 552/600 = 0.9200
  正确步骤召回率: 0.9857 (481/488)
  错误步骤召回率: 0.6339 (71/112)
  置信度差距: 0.5944 (正确 0.9774 - 错误 0.3830)

## beta_005 (beta=0.05, epochs=1)
  validate: accuracy=0.9688 (775/800), mean_margin=3.4510, chosen_reward=4.6676, rejected_reward=1.2167
  test: accuracy=0.9600 (384/400), mean_margin=3.4967, chosen_reward=4.6370, rejected_reward=1.1403

## beta_050 (beta=0.50, epochs=1)
  validate: accuracy=0.9712 (777/800), mean_margin=8.3313, chosen_reward=11.6431, rejected_reward=3.3118
  test: accuracy=0.9625 (385/400), mean_margin=8.4060, chosen_reward=11.5368, rejected_reward=3.1308

## beta_001 (beta=0.01, epochs=1)
  validate: accuracy=0.9688 (775/800), mean_margin=0.8779, chosen_reward=1.1161, rejected_reward=0.2382
  test: accuracy=0.9600 (384/400), mean_margin=0.8873, chosen_reward=1.1075, rejected_reward=0.2203

## d500 (epochs=5, samples=500)
test results (total=200, T=1.0):
  步骤准确率: 518/600 = 0.8633
  正确步骤召回率: 0.9857 (481/488)
  错误步骤召回率: 0.3304 (37/112)
  置信度差距: 0.3230 (正确 0.9596 - 错误 0.6366)

## d1000 (epochs=5, samples=1000)
test results (total=200, T=1.0):
  步骤准确率: 551/600 = 0.9183
  正确步骤召回率: 0.9836 (480/488)
  错误步骤召回率: 0.6339 (71/112)
  置信度差距: 0.5796 (正确 0.9701 - 错误 0.3905)

## de2 (beta=0.1, epochs=2)
  validate: accuracy=0.9712 (777/800), mean_margin=6.3311, chosen_reward=7.4658, rejected_reward=1.1346
  test: accuracy=0.9625 (385/400), mean_margin=6.4067, chosen_reward=7.4166, rejected_reward=1.0099

## de3 (beta=0.1, epochs=3)
  validate: accuracy=0.9712 (777/800), mean_margin=6.9373, chosen_reward=7.0862, rejected_reward=0.1490
  test: accuracy=0.9650 (386/400), mean_margin=7.0059, chosen_reward=7.0320, rejected_reward=0.0261

