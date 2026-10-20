# BaseRec

Python 3.11+ 的正反馈 LLM 推荐基线。没有数据库、embedding、向量库或 item memory；每位用户只维护一个字符串 memory。JSON memory 也序列化为这一个字符串，事件日志中的旧快照只用于审计和恢复，不作为额外推荐记忆。

## 运行

核心和默认 runner 仅依赖 Python 标准库。以下命令在 `model/BaseRec` 目录执行：

```bash
python3.11 -m unittest discover -s tests -v
python3.11 scripts/run_baserec.py \
  --input examples/tiny.json --output runs/tiny \
  --endpoint http://localhost:8000/v1 --model your-model
```

Endpoint 支持 `POST /chat/completions`、`response_format: {"type": "json_object"}`，返回 `choices[0].message.content` 和可选的 `usage`。API key 从 `OPENAI_API_KEY` 读取，可通过 `--api-key-env` 改用其他环境变量；不会写入输出文件。`--endpoint` 接受 API base URL 或完整 chat/completions URL。`--help` 列出全部选项。

原命令添加 `--resume` 可恢复；输入字节、模型、endpoint、memory 配置、tokenizer 和实现源码均须与原运行一致。

JSON sentence：

```bash
python3.11 scripts/run_baserec.py --input examples/tiny.json --output runs/sentences \
  --endpoint http://localhost:8000/v1 --model your-model \
  --memory-output-format json_points --json-point-format sentence
```

JSON preference theme 必须显式给出点数上下限：

```bash
python3.11 scripts/run_baserec.py --input examples/tiny.json --output runs/themes \
  --endpoint http://localhost:8000/v1 --model your-model \
  --memory-output-format json_points --json-point-format preference_theme \
  --json-min-points 1 --json-max-points 6
```

## 输入与协议

输入格式见 `examples/tiny.json`：`item_metadata` 为 item id 到元数据的映射，`histories` 为 user id 到按时间排序的 item id 数组，`candidates` 为每位用户的候选数组。JSON 对象键为整数字符串；历史与候选中的 id 必须为整数。历史至少一项，候选唯一且包含最后一项。单项历史直接测试，memory 保持空字符串。候选由调用方提供，本实现不采样负例。

调度按输入中的用户顺序，在每个 item position 轮流交错执行，只更新 `history[:-1]`。保留最后一次交互作为 held-out；若同一 item id 在更早的历史中真实出现，更早的出现仍是合法训练交互。

每次交互：

1. inference 仅看到旧 memory 和当前 item facts，输出三选一 affinity 与 rationale。
2. update 单独接收旧 memory 与当前已确认正反馈证据，完全不接收 inference 结果、历史列表、候选或 held-out 信息。
3. 新 memory 校验成功后提交；失败保留旧 memory。
4. 记录事件、前后 memory、统计、inference/update token 增量和包含失败重试的总增量。

inference / ranking 可见 title/name、地理位置、rating/stars、价格、类别、属性、营业时间。update 的字段白名单只有地理位置、类别、价格与非否定属性；不会传入评分、营业时间、item id 或商家名称。自由文本 description 不作为结构化证据。Prompt 明确禁止产生 dislike、拒绝、负反馈、名称、id 或推理过程；结构及长度是程序校验，生成内容的语义真实性仍取决于 LLM。

训练结束调用 `freeze()`，排序温度固定为 0，不给模型目标标签。非法、重复 id 被删除，缺项按原候选顺序补齐，返回目标的 1-based rank。NDCG 使用单个 held-out 相关目标，报告 NDCG@1/3/5/10 和 MRR。

## Memory 配置

- `text`：单段纯文本，默认最多 180 个以空白分隔的 words、8 个短句。`soft_hard` 不限制句子数。
- `json_points/sentence`：去重后的短句数组，默认 1–8 点。
- `json_points/preference_theme`：每点包含 `preference`、正整数 `evidence_count`、`confidence`（low/medium/high）与字符串数组 `contexts`。结构或去重后点数错误最多生成 3 次；证据计数由 LLM 基于旧 memory 与新正证据维护。
- `none`：无 token 限制。
- `soft`：只在 prompt 中要求 token 上限，不做事后 token 截断。
- `hard`：调用 token limiter 并验证结果没有超限。
- `soft_hard`：同时应用 prompt 与事后限制。

`--memory-token-limit-mode` 不为 `none` 时必须指定正整数 `--memory-token-limit`。默认 limiter 使用 UTF-8 字节数作为保守预算，不声称这是模型的精确 token 数；summary 中以 `utf8_byte_upper_bound` 标识。若需要精确的模型 tokenizer：

```bash
python3.11 -m pip install -e '.[tokenizer]'
# 在运行命令添加与你的模型匹配的 encoding，例如：
# --memory-token-limit-mode soft_hard --memory-token-limit 240 --tokenizer-encoding cl100k_base
```

JSON hard 限制会删除末尾 point，保持合法 JSON。若预算小到不能满足最小点数，会拒绝更新并按有限重试策略失败，不提交损坏 JSON。core 可注入实现 `count_tokens(text)` 和 `truncate(text, limit)` 的自定义 limiter。

## 输出与恢复

- `manifest.json`：输入 SHA-256、源码 SHA-256、完整协议及 fingerprint。
- `events/00000000.json`：每次已完成交互的原子事件记录，带 fingerprint、顺序号及记录哈希。
- `state.json`：model checkpoint、已完成交互数与累计 token 统计。
- `test_events/`：逐用户持久化的评测结果，恢复时不重复已完成排序。
- `interaction_trace.json`、`memory_history.json`、`final_memories.json`、`test_results.json`、`summary.json`：完成后的汇总输出。

runner 持有整个输出目录的非阻塞进程锁（POSIX/macOS/Linux）。先原子落盘事件，再写 checkpoint；恢复时校验事件连续性、协议与 checkpoint 前缀，并重放 checkpoint 后的事件。最终汇总可由日志重新生成。单交互最多尝试 3 次；每次 attempt 内 theme 的格式生成最多 3 次。排序也最多尝试 3 次。

原子写入使用同目录临时文件、fsync 和 replace。LLM 远端调用与本地日志无法构成同一事务：若进程在远端成功后、事件写入前崩溃，恢复可能重复该调用，且那次调用的服务端消耗无法从本地记录还原；用户 memory 不会重复应用一个已落盘事件。endpoint 未提供 usage 时，token 计数为 0，不进行猜测。

## Python API

```python
from src.models import BaseRec

model = BaseRec(llm, item_metadata, user_ids=[10])
record = model.run_interaction({"user_id": 10, "item_id": 1, "item_position": 0})
checkpoint = model.get_state()
model.set_state(checkpoint)
model.freeze()
result = model.rank_candidates(10, [1, 2, 3], target_item_id=3)
print(result["rank"])
```

`set_state` 接收完整 `get_state()` 结果；`apply_recorded_interaction` 要求记录的旧 memory 与当前 memory 完全一致，并且不能在冻结后调用。所有测试可离线执行，覆盖核心协议、三种输出、token 策略、事务失败、原子持久化、断点恢复、进程锁、HTTP 请求格式和指标计算。


## Experiment 1: Memory Value Identification

The frozen-memory evaluation reuses a single completed BaseRec snapshot; it does
**not** regenerate memory for each context. Formal evaluation requires the
matching training `manifest.json` next to `final_memories.json`, so the
dataset hash can be checked. A missing manifest fails closed; only exploratory
legacy snapshots should use `--allow-unverified-memories`.
Prepare a dataset using the same
train-prefix / held-out protocol as the original run:

```bash
# Train once. The affinity prediction before update is optional and not used
# by the memory updater, so it can be skipped for cost-sensitive runs.
python3.11 scripts/run_baserec.py \
  --input /path/to/dataset.json --output runs/memory \
  --endpoint http://localhost:8000/v1 --model your-model \
  --skip-training-inference

# Evaluate every context with the same frozen ranker and candidate permutations.
python3.11 scripts/run_memory_value.py \
  --input /path/to/dataset.json \
  --memories runs/memory/final_memories.json \
  --output runs/value \
  --endpoint http://localhost:8000/v1 --model your-model \
  --seed 42 --permutations 3
```

To resume an interrupted experiment, repeat the same command with `--resume`.
The runner checks metadata coverage **before** making any LLM requests. In
particular, `history_full` cannot run with zero usable training descriptions.
The output `input_audit` reports partial metadata coverage, empty train
prefixes, and users with repeated held-out items. Always inspect it before
interpreting the results.

Per-call results are saved in `runs/value/records`; summaries contain
mode-level NDCG/MRR, per-user metrics, response validity, context byte counts,
and inference usage when the endpoint supplies token metadata.

Supported modes (select a subset with `--modes`):
- `candidate_only`: empty user context, candidate facts still visible.
- `history_titles`: training-prefix item names/titles, in chronological order.
- `history_full`: names/titles plus descriptions from the same training prefix.
- `history_structured`: the exact allowlisted categories/geography/price/
  affirmative attributes visible to the BaseRec updater.
- `real_memory`: the frozen memory from that same user.
- `shuffled_memory`: another evaluation user's memory, using a deterministic
  derangement; requires two or more users.

All modes see identical candidate facts and identical candidate permutations.
`--include-candidate-description` supplies candidate descriptions to
**every** mode; the default preserves the original candidate-fact fields.
Incomplete or duplicate ranking outputs are rejected and retried by default,
rather than silently repaired. `--allow-repair` restores legacy behavior
only for diagnostics. `--resume` validates the experiment fingerprint and
saved-result hashes; changes to inputs, model, flags, or source require a new output directory.

**Scientific controls:** These modes are not all information-matched.
BaseRec's *updater* sees structured positive evidence but not titles/descriptions,
whereas `history_full` can see richer historical item text.
Use `history_structured` for the closest input-source control.
Control input budgets explicitly in a follow-up experiment rather than claiming
that any observed NDCG difference is solely caused by memory compression.
The study reports *inference* costs only: cost of generating/updating the saved
memories must be included separately for total-cost claims.
Hold out a separate validation interaction when tuning prompts or budgets;
do not repeatedly tune against the final test target.

The normal BaseRec training runner remains backward-compatible: training
inference is on by default and the existing `rank_candidates(user,candidates,target)`
API still works. Passing `user_context=` explicitly switches to the
common context-agnostic evaluation prompt. Update event logs now also include
`update_audit` (raw model output, normalized pre-limit output and
truncation details). Old run manifests are protected by source fingerprints;
do not attempt to resume an old run after changing the code.
