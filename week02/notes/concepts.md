# Week02 学习笔记 · LLM 基础概念地图

## 1. LLM 是什么？

**大语言模型（Large Language Model）**：基于 Transformer 架构、用海量文本训练、能够预测下一个 token 的概率分布的神经网络。

**本质**：`P(next_token | context)` 的条件概率模型。
**表现**：看起来"会理解、会推理、会创作"。

## 2. Transformer / Attention 直观理解

```
输入 tokens → [Embedding] → [Self-Attention ×N] → [FFN ×N] → 输出概率分布
                                ↑
                     每个 token 看其他 token 的"注意力权重"
```

**Self-Attention 解决什么**：长距离依赖。
- RNN/LSTM：信息逐层传递，长文本会"遗忘"
- Attention：任意两个 token 直接交互，O(n²) 但效果爆炸

**举例**：「那只猫躺在沙发上，它睡着了」
- "它" 通过 attention 知道指代"猫"而不是"沙发"

## 3. 训练三阶段

| 阶段 | 数据 | 目标 | 产物 |
|---|---|---|---|
| **Pretraining** | 全互联网文本（无标注） | 预测下一个 token | Base Model（会续写但不听话） |
| **Instruction Tuning (SFT)** | 人工写的指令-回答对 | 学会按指令回答 | Instruct Model（听得懂话） |
| **RLHF / DPO** | 人类偏好排序 | 学会"好"的回答 | Chat Model（说话让人舒服） |

## 4. Reasoning 模型 vs 通用模型

| 维度 | 通用模型（GPT-4o/DeepSeek-V3） | Reasoning 模型（o1/DeepSeek-R1） |
|---|---|---|
| 训练目标 | 直接生成答案 | 生成"思考过程"+答案 |
| 输出 | 短 | 长（含 CoT） |
| 擅长 | 写作、翻译、对话 | 数学、逻辑、代码、复杂推理 |
| 延迟 | 低 | 高 |
| 成本 | 低 | 高 |

**经验**：简单任务用通用模型，复杂推理才用 reasoning 模型。

## 5. 多模态模型

- 输入：文本 + 图像 + 音频 + 视频
- 输出：文本（部分模型也能输出图像/音频）
- 应用：合同扫描件理解、图表问答、视频内容审核

本周先用纯文本，多模态留到 week04。

## 6. Token / Context Window / Messages

```
用户输入 ─┐
          ├─→ messages (list of dict) ─→ Tokenizer ─→ token IDs ─→ Model
system ───┤        ↓
          │   每条 {role, content}
history ──┘        ↓
              token 总数必须 < Context Window
```

**Context Window**：模型一次能"看到"的 token 上限。
- GPT-3.5: 4K / 16K
- GPT-4o: 128K
- DeepSeek-V3: 64K / 128K
- Claude: 200K

**注意**：Context Window 是「输入 + 输出」总和，不是单独输入。

## 7. Prompt 工程核心思想

> Prompt 是「给模型看的程序」。

- 不是随便聊天，而是**精确规约**模型行为
- 越具体越好（角色、约束、格式、示例）
- 输出稳定性来自：明确指令 + Few-shot + 低温度 + 后处理校验

## 8. 为什么需要 RAG / Tool / Agent？

| 技术 | 解决的问题 | 本周对应 |
|---|---|---|
| **LLM 本身** | 自然语言理解与生成 | ✅ |
| **RAG** | 模型不知道"你公司的文档" | week05-08 |
| **Tool Calling** | 模型不能"查数据库、发邮件" | week03+week11 |
| **Agent** | 复杂任务需要"多步决策" | week14-16 |
| **Workflow** | 固定流程不需要"决策" | week18 |

**核心判断**：
- 一次性、纯文本生成 → 直接 Prompt
- 需要私有知识 → RAG
- 需要操作外部系统 → Tool
- 需要多步自主决策 → Agent

## 9. 本周踩坑总结

1. PowerShell 中文乱码 → `$env:PYTHONIOENCODING='utf-8'`
2. Mock 路由只看 user → 改为 system + user 都看
3. 8000 中文字超出 8K 窗口 → 直观理解了 RAG chunking 的必要性
4. 真实 LLM 会"偷懒"漏章节 → 后处理校验必须有
