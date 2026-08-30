# bench — Modal stream 压测

直打已部署（或本地）的 batch stream 接口，观察 **容量**：容器是否扩容、队列是否上涨、错误如何分类。

禁止把本目录挂进 pytest 或 `deploy.sh test`。一次压测会占 GPU、按墙钟计费。

## 打哪里

```
POST {BASE_URL}/translate/batch/image/stream/web
Content-Type: multipart/form-data
  images: 若干文件
  config: JSON 字符串
```

生产 Modal URL 形如 `https://<user>--manga-translator-web.modal.run`。本地可对 `http://127.0.0.1:8000`，但本地测不到 Modal 扩容。

响应是二进制帧流，与 `my-manga-translator` 的 `parseTranslationStream` 相同：

```
[1 byte status][4 bytes size BE][n bytes data]
```

| status | 含义 | 压测怎么记 |
|--------|------|------------|
| 1 | 过程：`image_completed:` / `image_failed:` / 其它日志 | `image_failed` 单独计数；请求可能仍继续 |
| 2 | 整批失败 | `stream_error`，结束本请求 |
| 3 | 排队位置 | **不是错误**，记 `queue_seen` / 排队时长 |
| 4 | 即将开始处理 | 与 status=3 配对，算 queue wait |
| 5 | 整批完成 | `stream_ok`，结束本请求 |
| 连接断开且未见 5 | 对现网 `stream closed before batch_completed` | `stream_error` |
| 超过 `--timeout` | 取消 reader | `timeout` |
| HTTP 非 2xx / 无 body | 提交失败 | `http_fail` |

## 发压模型

固定并发闭环，不是按 QPS 发射：

1. 启动 `concurrency` 个 worker。
2. 每个 worker：提交一批 → 堵住读 stream 直到结束/超时 → **立刻**再提交下一批。
3. 直到 `duration` 到时；不再发新请求，收尾 in-flight。

任意时刻 in-flight ≈ `concurrency`。并发单位是 **batch 请求**，不是图片张数。

对照当前 Modal 配置（`deploy/modal_config.py`）：

- `max_inputs = 2`：单容器最多 2 条并发 HTTP
- `min_containers = 0`：空闲会缩到 0，压测开头会有冷启动
- 生产 Workflow 的 `CONCURRENT_IMAGES = 2`：每发一批 2 张，与 `--batch-size 2` 对齐

建议阶梯（每次只改 concurrency，图片和 config 固定）：

```
phase 0: concurrency=1, duration≥2min   基线延迟、确认协议
phase 1: concurrency=2                  打满单容器 max_inputs
phase 2: concurrency=4 / 8              看扩容、排队、错误率
```

脚本只输出计数和延迟分位。容器数、GPU、OOM 看 Modal Dashboard / Grafana，按时间轴对齐。

## 目录约定

```
bench/
  README.md
  loadtest_stream.py      # 发压脚本（未实现时先看文件头注释）
  configs/                # 压测专用 config，不要复用生产保存配置
  fixtures/               # 固定图片集；也可指向 test/testdata 里已有图
```

`config` 必须满足服务端校验：

- `len(images) == len(config.image_identifiers)`
- **必须** `save.save_to = none`。省略 `save` 会默认 `local` 写盘；`supabase_storage` 会刷对象存储。压测不需要真正存图。
- translator / detector / ocr 选型当作实验维度，一次压测内固定，否则 Grafana 说不清是负载还是模型变了。

每发请求应改写 `image_identifiers`（带请求 id），避免多 worker 撞同一标识。

## 怎么跑

在 `manga-image-translator` 仓库根目录：

```bash
python bench/loadtest_stream.py \
  --url https://<user>--manga-translator-web.modal.run \
  --images bench/fixtures \
  --config bench/configs/no_save.json \
  --concurrency 2 \
  --batch-size 2 \
  --duration 60 \
  --timeout 180
```

| 参数 | 含义 |
|------|------|
| `--url` | Modal 或本地 base URL，不要带 path |
| `--images` | 图片目录；按 `--batch-size` 切片，不足则循环复用 |
| `--config` | 压测 config JSON |
| `--concurrency` | 同时 in-flight 批次数（真正的压测旋钮） |
| `--batch-size` | 每发几张，建议先对齐生产的 2 |
| `--duration` | 墙钟秒数 |
| `--timeout` | 单次 stream 上限秒数。现网 `parseTranslationStream` 没有超时，脚本必须自带，否则 hang 时 Grafana 会看起来空闲 |

跑之前确认：目标环境、config 未开 supabase 保存、Grafana / Modal 面板已打开。跑完看脚本错误分类 + 平台负载，不要只看其中一个。
