## 翻译样例分析

### 测试集样例

| 序号 | 中文原文 | 英文参考翻译 |
|------|---------|-------------|
| 1 | 记录指出 HMX-1 曾询问此次活动是否违反了该法案。 | Records indicate that HMX-1 inquired about whether the event might violate the provision. |
| 2 | 该指挥官写道“我们问的一个问题是这是否违反了《哈奇法案》，并被告知没有违反。” | "One question we asked was if it was a violation of the Hatch Act and were informed it was not," the commander wrote. |
| 3 | “听起来你被锁住了啊，”副司令回复道。 | "Sounds like you are locked," the Deputy Commandant replied. |
| 4 | 白宫将此次“美国制造”活动定义为官方活动，因此不受《哈奇法案》管辖。 | The "Made in America" event was designated an official event by the White House, and would not have been covered by the Hatch Act. |
| 5 | 但是即使是官方活动也带有政治色彩。 | But even official events have political overtones. |

### 模型翻译对比

**注意**：由于需要加载模型进行推理，翻译样例的生成需要运行推理脚本。
可以使用以下命令生成翻译样例：

```bash
# RNN模型翻译
python inference.py --model_type rnn --checkpoint checkpoints/rnn_best.pt --input "记录指出 HMX-1 曾询问此次活动是否违反了该法案。"

# Transformer模型翻译
python inference.py --model_type transformer --checkpoint checkpoints/transformer_best.pt --input "白宫将此次"美国制造"活动定义为官方活动，因此不受《哈奇法案》管辖。"

# T5模型翻译
python inference.py --model_type t5 --checkpoint checkpoints/t5_best --input "但是即使是官方活动也带有政治色彩。"
```
