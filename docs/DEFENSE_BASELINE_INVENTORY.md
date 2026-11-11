# 防御基线与复现证据清单

更新日期：2026-11-11。论文元数据见[文献记录](LITERATURE.md)。表中区分论文方法、公开源码审计、组件一致性和项目适配器；不把这些状态统称为完整复现。

| 方法 | 证据与当前实现范围 | 尚不能作出的结论 |
|---|---|---|
| SemBind | [官方masker组件比对](../reports/repro_sembind_masker_20261103/report.md)；[官方反演组件检查](../reports/sembind_official_audit_20261110/report.md)；[项目适配器完整预算结果](../reports/full_imprint_20261111/report.md) | 有限组件一致不等于完整原生方法复现；开发语义码相近不证明公开特征绕过或普遍碰撞 |
| PGID | [历史n=1及n=5原始CSV核验](../reports/repro_pgid_20261103/report.md)，各检查点防御后均零接受；模型快照的官方字节等价性未建立 | 小样本零接受不证明完整论文复现或总体安全性；BA上升但未过门限不证明防御失守 |
| Geometry | [公开评估契约](EVALUATION_CONTRACT.md)及[公式重实现诊断](../reports/repro_geometry_paper_task_20261103/report.md) | 未完整确定局部映射与拒绝规则；项目L-SPD结果不能充当作者原生组件失效证据 |
| SEAL | [固定源码契约审计](SEAL_EVALUATION_CONTRACT.md)，区分语义路径和穷举路径 | 测试脚本存在不证明原生语义防伪评测完成；现有GS样本不能替代原生SEAL参考 |
| PAI | [最终论文与固定源码契约](PAI_EVALUATION_CONTRACT.md)，普通验证明确、完整稳健验证仍有材料缺项 | 无法把GS参考图直接输入为PAI原生水印；源码审计不等于完整模型结果 |
| ISTS | [CVPR2026 Findings论文](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Towards_Robust_Content_Watermarking_Against_Removal_and_Forgery_Attacks_CVPRF_2026_paper.html)与作者仓库已定位 | 尚未建立本项目同条件完整比较；参数空间大小本身不证明可被伪造 |
| CoverLock | [官方组件与项目适配说明](../reports/content_verifier_20261110/report.md) | 辅助图GS适配不等于原生后处理防御 |
| SLICE、Sign Clamping、PP-Mark、GS++ | 文献来源及状态见[LITERATURE.md](LITERATURE.md) | 尚未完成同条件原生复现，不据摘要进行数值排名或断言漏洞 |

## 项目验证器

[六条件确认](../reports/geometry_confirmation_20261111/report.md)中全组几何恢复63张合法裁剪查询；[官方九条件](../reports/gs_published_conditions_20261111/report.md)中无新增恢复且损失2张默认接受图；[完整Imprint](../reports/full_imprint_20261111/report.md)中没有额外伪造拒绝收益。[自然负例](../reports/unwatermarked_validation_20261111/report.md)中默认与几何均零误接受。没有证据支持将当前原型写成全面超越上述防御的方法。

## 公平比较要求

生成端方法使用各自原生生成与验证流程，检测端方法记录其实际可见信息。统一数据分母、目标信息权限和已发表评测预算；保留失败的合法参考，不做条件分母筛选。分别报告实际合法检出率、伪造接受、错误身份和未加水印负例、图像质量及计算成本。零相关误报观察不是极低FPR保证。

本项目只使用已发表顶会顶刊的既定攻击基线，不设计或改进攻击。已有小样本、短预算和数值模拟不能替代完整比较。
