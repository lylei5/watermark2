# 扩散模型语义水印防御汇报

日期：2026-11-13。基于用户提供的中山大学10页PPT模板，保持原页面尺寸、校园图片与校徽。演示文稿包含逐页讲稿与证据链接，表格为可编辑对象。

- [演示文稿](output/语义水印防御研究汇报_中山大学_20261113.pptx)
- [逐页讲稿与来源](content.json)
- [编辑脚本](../../scripts/presentation/update_defense_briefing.mjs)

汇报依次介绍防御目标、SemBind机制、三项基线的已验证范围、PGID完整预算结果、SEAL语义规则的良性处理结果、内容恢复区域重叠，以及拟研究的实例绑定方案。

实验结果属于已核验的开发证据。按论文补齐的实现、作者原代码、正常图链路验证与完整论文复现分别说明。当前没有建立新防御优于SemBind、PGID或SEAL的结论，内容密钥方案尚待实现和独立验证。

主要证据见[三个基线报告](../../reports/three_baselines_20261113/report.md)、[半径诊断](../../reports/phase1_hamming_radius_20261113/report.md)及[内容密钥可行性评估](../../docs/CONTENT_KEY_FEASIBILITY.md)。桌面源文件SHA256为`30fe1cebaaa3c7d31a492e3a834494d348024a2edfee2225e10c5c3eee5c85ac`。
