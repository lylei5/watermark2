# 三个防御基线的来源和执行记录

已固定SemBind `337c9f90a150157ac1d410208ae20e6530ca8cdd`、PGID `dc4ad963ffefcb4cf84340084db617472983d3e0`、SEAL `92d31b31b93a6e373fe88a584c640beafb68c6fa`。三个干净源码副本均位于249的`third_party/frozen`，不改动上游数值或攻击代码。验收定义见[执行契约](../../docs/THREE_BASELINE_REPRODUCTION.md)，动态状态见[project_state.json](../../project_state.json)。

[源码及权重对照](source_equivalence.json)验证PGID新旧22个Python文件的可执行AST一致，忽略注释和文档字符串；该检查是来源等价核验，不是新的模型实验。SemBind现用的inversion.py、model.py及src下三个生成/反演文件与官方归档逐字节一致。完整水印绑定入口仍未公开，论文补齐部分不能宣称是作者原码。

[SEAL环境与资产](seal_preflight.json)已完成249上的CPU导入和全部本地权重文件哈希。BLIP2两片权重、spiece以及作者caption-embedding权重均与Hugging Face上游LFS哈希一致。扩散模型暂用固定Manojb SD2.1镜像；stabilityai原始fp16快照同一性未认证，公开元数据请求返回401。未将此限制写成已证实模型不同。

SEAL语义入口使用官方生成、caption、SimHash和反演函数；最终布尔判定补自ICCV最终算法3和补充§9.1：1024块、每块7位、距离严格小于2.3、至少12块匹配。原入口使用所载模型的PNDM生成调度器，反演用作者DDIM代数及该调度器时间步；不擅自替换生成调度器。公开代码没有暴露论文secret salt，保留这个协议差异。

100提示词（COCO/SDP各50）开发链路已冻结于[协议](../../protocols/seal_native_20261113_v2.json)，不是论文1000图实验或独立确认。协议SHA256 `a249befa285e8033358cc0b0c228c9c48cba6438ceaf04cb74e31f8ec6f1a490`。每例保存预览、合法图、无水印图、候选caption、反演潜变量、全部1024个距离和布尔判定。无攻击、无良性扰动结果，不能据此宣称完整防御性能。

初次协调器因服务器Python不提供os.pidfd_open，在任何GPU工作进程启动前退出。原运行与[原协议](../../protocols/seal_native_20261113.json)保持冻结；v2仅改为该主机支持的Linux pidfd系统调用等待退出事件，没有轮询进度、改变模型或改变参数。
