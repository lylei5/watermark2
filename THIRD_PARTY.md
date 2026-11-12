# 第三方论文代码与本地证据

目录历史核验日期：2026-11-11；新增干净源码核验日期：2026-11-13。下表记录Git子模块固定版本及249服务器实际目录状态；源码存在、权重可用、组件验证与完整方法复现分别判断。论文元数据见[文献记录](docs/LITERATURE.md)。

## 当前原生基线的干净源码

249的`third_party/frozen`中固定三个独立源码目录：

| 目录 | 上游固定提交 |
|---|---|
| `SemBind_337c9f9` | XinZhang1999/SemBind `337c9f90a150157ac1d410208ae20e6530ca8cdd` |
| `PGID_dc4ad96` | MinhQDuong/PGID `dc4ad963ffefcb4cf84340084db617472983d3e0` |
| `SEAL_92d31b3` | Kasraarabi/SEAL `92d31b31b93a6e373fe88a584c640beafb68c6fa` |

来源归档、逐文件哈希与新旧版本核验见[来源证据](reports/three_baselines_20261113/source_equivalence.json)及[清单](reports/three_baselines_20261113/source_manifest.json)。下列原目录检查保持其原始日期含义，不能代替上述新运行的冻结来源。

## 仓库子模块

| 路径 | 上游与固定提交 | 论文与用途 |
|---|---|---|
| `third_party/semantic-forgery` | [and-mill/semantic-forgery](https://github.com/and-mill/semantic-forgery)，`ca68950cd7b53b8438a49580f134b2a0db28d778` | [CVPR2025，2412.03283](https://arxiv.org/abs/2412.03283)；GS、Tree-Ring及已发表Imprint/Reprompt评测入口 |
| `third_party/geodistort-wm` | [chengyi-chris/geodistort-wm](https://github.com/chengyi-chris/geodistort-wm)，`72880c9bb1cee0b04bcb2433deb2ab2cc95ff622` | [ICML2026，2606.29807](https://arxiv.org/abs/2606.29807)；公开评测入口，不等于已公开完整几何防御函数 |

`.gitmodules`与Git索引中的两个固定提交相符；服务器两个对应目录的HEAD和origin相同。

## 服务器第三方目录

根目录为`/data0/liyanlei/watermark_research/third_party`。这是一次源码状态检查，不是权重或运行环境完整性认证。

| 目录 | 来源与本次观察 | 本项目证据 |
|---|---|---|
| `SemBind` | [XinZhang1999/SemBind](https://github.com/XinZhang1999/SemBind)；目录存在，本次未取得可用Git元数据。历史契约固定版本为`337c9f90a150157ac1d410208ae20e6530ca8cdd`，不能把它当作今天目录HEAD的证明 | [官方源码契约](docs/EVALUATION_CONTRACT.md)、[组件审计](reports/sembind_official_audit_20261110/report.md)；生成绑定链路为项目适配 |
| `PGID` | [MinhQDuong/PGID](https://github.com/MinhQDuong/PGID)；HEAD `eb7fb0ddc46a07e4f52a0a79038e546f7c925b50`，origin一致 | [n=1与n=5历史实验核验](reports/repro_pgid_20261103/report.md)；模型快照等价性未建立 |
| `SEAL` | [Kasraarabi/SEAL](https://github.com/Kasraarabi/SEAL)；HEAD `92d31b31b93a6e373fe88a584c640beafb68c6fa`，origin一致 | [原生语义与穷举验证契约](docs/SEAL_EVALUATION_CONTRACT.md)；ICCV2025 |
| `ISTS` | [hala64/ISTS](https://github.com/hala64/ISTS)；目录存在，本次未取得可用Git元数据 | [基线清单](docs/DEFENSE_BASELINE_INVENTORY.md)；不声明完整复现 |
| `PAI` | [QingyuLiu/PAI](https://github.com/QingyuLiu/PAI)；本次未见该目录 | 仓库保存的是[固定源码审计契约](docs/PAI_EVALUATION_CONTRACT.md)及小型来源证据；不声明服务器已部署完整模型 |

HEAD相符不证明工作树未改、依赖相同或实验与论文等价。需要这些条件的实验应另存源码哈希与环境记录。已有短预算结果保留为历史诊断；后续评测使用已发表顶会顶刊基线的既定实现与完整预算。
