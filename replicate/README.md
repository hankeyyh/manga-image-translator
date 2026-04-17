# Replicate 部署说明（V1）

本目录用于将项目的默认离线链路部署到 Replicate，且不修改现有业务 pipeline。

## V1 输入输出契约

- 输入：
  - `image`（必填）
  - `target_lang`（默认 `ENG`）
  - 可选覆盖：`detector`、`ocr`、`inpainter`、`translator`
- 输出：
  - 单张 PNG 图片（翻译后的最终结果）

`predict.py` 直接复用 `MangaTranslator`，并保留默认组合：
- detector: `default`
- ocr: `48px`
- inpainter: `lama_large`
- translator: `sugoi`

## 模型目录

- 默认：`/src/models`（通过 `MANGA_TRANSLATOR_MODEL_DIR` 控制）
- 本项目统一通过 `ModelWrapper._MODEL_DIR` 指向模型路径

## 权重策略（V1 选型）

V1 默认采用 **构建期预取**（镜像更大，但首请求更快、更稳定）。

- `cog.yaml` 在 build 阶段执行 `python -m replicate.prefetch_models`。
- 默认链路的完整权重目标（`model_dir` 下）：
  - `detect-20241225.ckpt`
  - `ocr_ar_48px.ckpt`
  - `alphabet-all-v7.txt`
  - `lama_large_512px.ckpt`
  - `jparacrawl/spm.ja.nopretok.model`
  - `jparacrawl/spm.en.nopretok.model`
  - `jparacrawl/big-ja-en`（目录）
  - `jparacrawl/big-en-ja`（目录）
  - `sugoi/spm.ja.nopretok.model`
  - `sugoi/spm.en.nopretok.model`
  - `sugoi/big-ja-en`（目录）

如需临时切回懒加载，可跳过预取步骤，仅依赖运行时按需下载。

## 本地验证（建议）

### 1) 语法与依赖检查

```bash
python -m py_compile predict.py replicate/prefetch_models.py replicate/validate_v1.py
```

### 2) 单图 smoke test + 延迟记录

```bash
python -m replicate.validate_v1 --image <your_image_path> --runs 3
```

输出包含：
- 每轮耗时（秒）
- 平均耗时 / P95 估算
- 使用配置（detector/ocr/inpainter/translator）

## 发布与观察（运行手册）

1. 登录并初始化：
   - `cog login`
   - `cog push r8.im/<owner>/<model-name>`
2. 发布首个私有版本后，记录：
   - 版本号
   - 硬件规格（建议 L4，备选 T4）
   - 首次请求耗时、热启动耗时
3. 发布后 24 小时观察（自动统计）：
   - `python -m replicate.observe_v1 --owner <owner> --model <model-name> --version <version-id> --hours 24`
   - 如不指定 `--version`，会聚合同模型最近窗口的所有版本请求
   - 使用 `REPLICATE_API_TOKEN` 鉴权（或 `--api-token`）
4. 重点看：
   - 失败率（`failure_rate`）
   - 超时率（`timeout_rate`）
   - OOM 率（`oom_rate`）
   - 平均耗时与 P95（`latency_sec.avg/p95`）
5. 二期能力准入（默认建议）：
   - 样本量 `>= 30`
   - OOM 率 `<= 1%`
   - 超时率 `<= 2%`
   - 总失败率 `<= 3%`
6. 二期再考虑：
   - 批处理
   - 流式进度
   - GPT/Gemini 等密钥型翻译器
