# 100对确认资产准备完成：Flickr字段修正与完整数据审计

**100张COCO载体、100条唯一caption及100个固定配对已准备并封存。** 200个COCO源图ID及200个Flickr photoID互不重叠；实际下载100张载体，历史/组内去重拒绝0。caption只取元数据，不下载其来源图片。没有生成参考图、水印秘密或认证凭据，没有模型/GPU或目标评分。资产完成不等于100对模型实验完成。

权威清单：[manifest.json](manifest.json)，SHA256 **`a41bbad8a755418cbc06f83a6660f170dd0b48d5b9cb31d31d434d42657c8220`**。实际图片及RGB512 PNG留在服务器`/data0/liyanlei/watermark_research/datasets/confirmation100_relay_v11/`，GitHub只发清单、协议、来源/许可、哈希和审计，不发布原图、PNG或完整注释包。

## v1→v1.1修正

[COCO官方schema](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/format-data.htm)只规定`flickr_url`是字符串，未保证作者页路径。[Flickr官方URL文档](https://www.flickr.com/services/api/misc.urls.html)区分静态图片URL与包含用户ID的照片页，并说明旧静态URL形式。真实COCO2017val 5,000条记录全部是静态图链接，能解析photoID但不含作者账户。[v1失败](../confirmation100_assets_relay_v1/report.md)是我们错误的字段假设，不是外部资产不足。

v1.1支持严格校验的legacy `farm*.staticflickr.com`、现代`live.staticflickr.com`及`flickr.com/photos/account/photoID`形式；保留原URL、COCO ID、photoID和原许可URL。静态源的作者账户/显示名记null，不虚构归属。缺失/非法来源或无法提取photoID按预定规则排除，不人工挑图。原CC BY2.0/3.0/4.0白名单、seed、SHA256排名、历史覆盖清单、排除阈值、100张数量和RGB512预处理不变。[33行选择器差异](../../protocols/confirmation100_v11_selector.diff)、[冻结v1.1协议](../../protocols/confirmation100_assets_v11.json)可审查，协议SHA256 `3d89100585bacc14bffba10423d46a2d62c712fbc7d3a634d0b758c9f1496f7f`。

实际许可池857张全部为COCO元数据声明CC BY2.0；其他4,143张仍被同一白名单排除。元数据许可声明与真实当前权利并非同一证明。100张载体的作者账户均未知，全部标`not_cleared_do_not_publish_image`；只在已批准内部研究中使用，当前不具备重新分发原图条件。来源清单和caption原文/注释CC BY4.0归属保留；没有拿仓库代码许可代替图片许可。

## 获取与交接修复

[完整标注包取得链路及双端校验](../confirmation100_assets_relay_v1/report.md)保持原证据：252,907,541字节，SHA256与TFDS一致，ZIP全部6成员CRC通过；仅3,872,473字节`captions_val2017.json`及其机械子集容器中转。本轮复用这些校验，不重复下载或复制完整252MB包。单图通过正常TLS的官方S3路径式入口获取，每张Windows与服务器原图SHA256一致。

网络采集监督PID21140运行631.694109秒后在最后一张就绪JSON交接处退出1；[原失败](../confirmation100_assets_relay_v11/report.md)完整保留。SCP直接创建最终就绪文件名存在竞态，不把“文件出现”当作传输已完成。单独[冻结离线完成计划](../../protocols/confirmation100_v11_completion.json)在收据完整、所有100图已收到后复用原材料：验证body大小/SHA后通过独占原子链接发布不可覆盖的就绪记录，仍保留staging。未重发任何单图请求。

离线完成退出0，用时36.428302秒；逐文件证明302个既有数据文件字节未变，仅补齐最后一张的规范原图路径和PNG，未改采样算法或此前图片。[保留核验](prior_data_preservation.json)、[仅物化差异](../../protocols/confirmation100_v11_completion.diff)可复核。旧网络采集器保留为关闭失败档案；当前基线应读取已封存清单，不重新执行它。

## 验证与边界

8项字段测试在Windows和服务器通过，覆盖静态/作者页/缺字段/非法URL；5项中转边界测试通过。原子收据4项测试在实际Linux接收端通过。Windows沙箱硬链接权限使其中1项未通过，未在Windows重试该操作；原子发布实际发生于Linux，不声称Windows同样验证成功。[CPU记录](cpu_checks.json)保留。

服务器[独立规则重放](selection_replay_verification.json)核对857来源池、完整cover/caption顺序、200来源身份、100原图和100 RGB512 PNG实际哈希/像素以及Windows收据；全部通过。公开[复核](public_verification.json)核对315项终端封存、协议/源哈希、许可/归属、配对及收据绑定。没有人工作品内容筛选、目标分数或原始噪声/密钥输入。

原seed `wm-native100-20261025-v1`与579历史图片、2626份元数据的排除库不变。0次排除不证明完全未用于历史开发：裁剪/转码、未记录或项目外使用可能未覆盖，也不证明基础模型训练未见过。这100对专用于冻结基线确认，不用于新方法或超参数选择；未来方法仍需独立确认。

## 预算及下一步

按[两端存储审计](storage_audit.json)，计入Windows完整包、保留失败部分文件、100原图及日志、服务器旧/新所有确认数据副本，合计423,966,994字节，低于512MiB。工作区加服务器研究文件保守合计2,317,623,264字节，低于10GiB，排除既有cache/env/git。重传/数据流校验的传输与落盘副本分别记录，网络协议开销未测量。

下一项是审查完整原生更正版Imprint基线协议、独立私有GS材料及实际资源预算，再决定100对模型准入。此次没有启动该模型实验；Jain原SD1.4资产仍缺，阶段30与八攻击对照继续hold。阶段31四对校准及事后LPIPS结论不变，没有新增SOTA或方法新颖性结论。
