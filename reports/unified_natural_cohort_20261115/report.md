# 统一自然图负例：40000图队列冻结

**共同自然图资产已完成获取和审计：40670张全部可解码，筛选后保留40642个候选簇，固定20000张用于阈值校准、20000张用于独立FPR验证，另有642个簇未使用。** 两个集合的来源ID不重叠，40000张的预处理像素哈希和pHash均唯一。这是数据准备结果；尚未产生统一工作点的防御排名。

| 审计项 | 实测结果 |
|---|---:|
| 官方COCO test2017原始图像 | 40670 |
| 无法解码的图像 | 0 |
| 对照的外部开发图像 | 5008 |
| 全部有效图上的pHash近重复边 | 38 |
| 因与外部开发簇关联而排除的test2017图 | 3 |
| 去重并排除开发关联后的候选簇 | 40642 |
| 阈值校准集 / 独立自然负例验证集 | 20000 / 20000 |
| 两集合的COCO来源ID交集 | 0 |
| 选中图像的不同RGB512哈希 / pHash | 40000 / 40000 |
| 未使用候选簇 | 642 |
| CPU图像与重复审计耗时 | 283.59秒 |

## 数据来源和完整性

数据来自[COCO官方test2017归档](http://images.cocodataset.org/zips/test2017.zip)，通过其官方S3对象传输。归档共6646970404字节，793个8MiB分块（末块不足8MiB），重组后的multipart ETag与官方值一致：`1d4d2301bdee781394c21718428764d2-793`。全部JPEG成员在解压时通过ZIP CRC检查。

归档SHA-256为 `c7908c3c9f94ba2f3340ebbeec58c25db6be8774f18d68c2f15d0e369d95baba`。分块、归档和图像均保存在249；本机只作固定公开对象的内存流式转发，不保存图像归档，也没有运行数值实验。传输辅助进程已在完成后关闭。

原始归档和图像：`/data0/liyanlei/watermark_research/datasets/coco_test2017_official`。冻结数据清单与全部特征：`/data0/liyanlei/watermark_research/datasets/unified_coco_nulls_20261115`。

## 划分与去重规则

所有图像统一转RGB，再用Pillow的ImageOps.fit处理为512×512，BICUBIC、中心裁剪，不额外旋转EXIF。保存原文件SHA-256、预处理像素SHA-256、尺寸和64位DCT pHash。完全相同像素及pHash汉明距离不超过4的图像合并为传递闭包簇；每簇至多选一张。与5000张现有COCO val2017及8张耗时开发图形成同一簇的test2017图全部排除。这些外部资产仅被读取。

另用COCO val2017提示词元数据验证test2017来源ID无交集。对剩余簇按预先指定的 `SHA256(watermark2-null-20261115/ + source_id)` 排序，前20000个为校准集，随后20000个为验证集。分割不依据任何防御分数。公开[来源ID清单](cohort_ids.json)，以便各方法严格复用同一划分；完整输入清单SHA-256为 `13c09dd26845ccad25d9e553888b60fb2e3283de6fd89aaee00e9508b96917a6`。

感知哈希筛查不能保证发现全部相关照片。本报告中的分离指项目已审计来源、像素和近重复规则；不声称这些图像未被基础模型预训练见过，也不把COCO上的FPR推广到所有自然图分布。

## SEAL自然图提取协议

已冻结[40000图SEAL协议](../../protocols/unified_seal_nulls_20261115.json)，SHA-256为 `1a3283f12074a95cbfede7a2bb64095dfa7cb812c65c00cceb363ca34d2de20a`。运行目录为 `/data0/liyanlei/watermark_research/runs/unified_seal_nulls_20261115`，实际状态以[project_state.json](../../project_state.json)和服务器完成回执为准。

使用原SEAL caption、语义编码和反演函数，50个请求步数、原PNDM时间步网格，逐图核对51次UNet调用。论文未公开的私有salt采样按既有补齐实现执行，保留明确的SHA-256序列化与PRNG约定；不声称与作者未公开实现逐字节一致。一个新随机provider salt在推理前固定，因此结果描述此身份下的条件FPR。

每块距离阈值固定2.3，只在校准集上选取匹配块数的整数阈值，分别对应不超过1%和0.1%的可达经验工作点；随后在独立验证集报告实际FPR及Clopper–Pearson 95%区间。原论文的至少12块工作点另报。不得依据验证集或后续攻击重新调阈值。

提取保存逐图输入与产物哈希、caption、恢复潜变量、参考潜变量、1024块距离、完整调用数、耗时和显存。支持按相同协议和逐图哈希校验续跑；任何失败都会保留并使完成检查失败，不静默减少40000图分母。最多使用三张当前空闲GPU，按UUID绑定且排除GPU6。尚未完成的合法TPR、攻击成功率和空间篡改检测不能由本阶段替代。

[启动验收](seal_startup_validation.json)已核验三个分片各自首张图：GPU7/5/4，输入及产物哈希一致，潜变量和1024块距离均有限，每图51次UNet调用。该验收只证明实际推理已开始，不代表40000图完成，也未查看验证集FPR。

## 证据与后续

[归档回执](archive_receipt.json)、[重复与来源审计](audit.json)、[环境和清单校验](environment.json)、[SEAL模型哈希清单](seal_model_manifest.json)均来自249。获取与CPU审计的冻结执行目录为 `/data0/liyanlei/watermark_research/runs/unified_data_pipeline_20261115`。

后续需要完成SemBind与PGID的共同载体/分数接口、同一负例队列上的完整提取及独立FPR验证，再填入合法图、逐攻击、质量与开销结果。实例密钥恢复保持冻结；本阶段没有生成攻击或开展新机制实验。
