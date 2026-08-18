# compare_config_logs

从两份 **verl** 训练日志里提取 TaskRunner 打印的完整 config，拍平后生成 Excel 对比表。适合 NPU / GPU、不同 batch、不同镜像之间的配置对齐。

## 使能方法

在能读到两份日志的环境里执行即可，不依赖 NPU，也不用进训练容器。

在仓库根目录执行：

```bash
# 1. 进入工具目录
cd 01-tools/02-verl/compare_config_logs

# 2. 安装依赖（只需一次）
pip install -r requirements.txt

# 3. 传入两份 verl 日志
python compare_config_logs.py \
    npu.log \
    gpu.log \
    -o config_compare.xlsx \
    --name-a npu \
    --name-b gpu
```

最短用法（列名用文件名，xlsx 写到当前目录）：

```bash
python compare_config_logs.py log_a.out log_b.log
```

参数：

| 参数 | 必填 | 说明 |
|------|------|------|
| `log_a` | 是 | 日志 A |
| `log_b` | 是 | 日志 B |
| `-o` / `--output` | 否 | 输出 xlsx；默认 `config_compare_<name_a>_vs_<name_b>.xlsx` |
| `--name-a` | 否 | Excel 列名 A，默认用日志文件名 |
| `--name-b` | 否 | Excel 列名 B，默认用日志文件名 |

成功时终端类似：

```
npu.log: parsed OK (end~L1234), top-level keys=...
gpu.log: parsed OK (end~L5678), top-level keys=...
Wrote config_compare.xlsx
total=... same=... diff=... only_a=... only_b=...
```

## 日志要求

脚本找的是 verl / Ray `TaskRunner` 把 Hydra config `pprint` 出来的那一段 Python dict，不是任意训练 stdout。

识别范围：

- 起点：出现 `{'actor_rollout_ref'`
- 终点：出现 `'transfer_queue'`，且花括号配平

两类日志都能解析：

1. **带 Ray 前缀**：每行类似 `(TaskRunner pid=1234) 'key': value`
2. **纯 pprint**：例如 NPU 重定向的 `.out`，没有 TaskRunner 前缀

日志里可以夹杂 `UserWarning`、`INFO` 等噪声，脚本会跳过。找不到这段 dict 会报：

```
RuntimeError: Config dict not found in ...
```

常见原因：任务还没跑到 config dump、日志被截断、或不是 verl `main_ppo` 这类日志。

## Excel 说明

三个 sheet：

| Sheet | 内容 |
|-------|------|
| 全部配置比对 | 两边所有拍平后的 key |
| 差异项 | 只保留不同 / 仅一侧存在的项 |
| 汇总 | key 总数、相同数、差异数、仅 A、仅 B、两个日志路径 |

列：

| 列 | 含义 |
|----|------|
| key | 拍平路径，如 `actor_rollout_ref.rollout.gpu_memory_utilization` |
| `<name-a>` | 日志 A 的值，缺失为 `<MISSING>` |
| `<name-b>` | 日志 B 的值，缺失为 `<MISSING>` |
| 状态 | `相同` / `不同` / `仅<name-a>` / `仅<name-b>` |

颜色：绿=相同，黄=不同，橙=仅 A，蓝=仅 B。表头可筛选。

嵌套 dict / list 会拍平：

- dict：`a.b.c`
- 含结构的 list：`a[0].b`
- 全是标量的 list：整段保留为一个值

## 限制

- 只对比 **config dump**，不对比 loss、吞吐、NPU 算子日志。
- 值按字符串比较（`repr`），`1` 和 `1.0` 会判为不同。
- 路径类配置（model path、log dir）几乎总会进「差异项」，这是环境差异，不一定是训练配置写错。
- 需要日志里完整打出 config；中途 kill 且 dump 不完整时解析会失败。
