# Replicate 发布检查清单

## 发布前

- [ ] `python -m py_compile predict.py replicate/prefetch_models.py replicate/validate_v1.py`
- [ ] 选择权重策略（V1 默认构建期预取）
- [ ] 已执行 `python -m replicate.prefetch_models`（或确认由 `cog.yaml` build 自动执行）
- [ ] 本地已跑 `python -m replicate.validate_v1 --image <sample> --runs 3`
- [ ] 记录基线：首轮耗时、平均耗时、P95、是否 OOM

## 发布

- [ ] `cog login`
- [ ] `cog push r8.im/<owner>/<model-name>`
- [ ] 记录 Replicate 版本号与发布时间
- [ ] 标注硬件规格（建议 L4，备选 T4）
- [ ] 确认模型是私有可见性，记录 owner/model/version

## 发布后观察

- [ ] 记录冷启动与热启动差异
- [ ] 执行 `python -m replicate.observe_v1 --owner <owner> --model <model-name> --version <version-id> --hours 24`
- [ ] 记录指标：失败率、超时率、OOM 率、平均耗时、P95
- [ ] 对照阈值：OOM<=1%，超时<=2%，失败率<=3%，样本量>=30
- [ ] 若阈值未达标，先修稳定性（显存/并发/超时）再做二期
- [ ] 若阈值达标，进入二期（批处理、流式进度、更多翻译器）
