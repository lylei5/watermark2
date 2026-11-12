# 三个防御基线的来源和执行记录

已固定SemBind `337c9f90a150157ac1d410208ae20e6530ca8cdd`、PGID `dc4ad963ffefcb4cf84340084db617472983d3e0`、SEAL `92d31b31b93a6e373fe88a584c640beafb68c6fa`。三个干净源码副本均位于249的`third_party/frozen`，不改动上游数值或攻击代码。验收定义见[执行契约](../../docs/THREE_BASELINE_REPRODUCTION.md)，动态状态见[project_state.json](../../project_state.json)。

[源码及权重对照](source_equivalence.json)验证PGID新旧22个Python文件的可执行AST一致，忽略注释和文档字符串；该检查是来源等价核验，不是新的模型实验。SemBind现用的inversion.py、model.py及src下三个生成/反演文件与官方归档逐字节一致。完整水印绑定入口仍未公开，论文补齐部分不能宣称是作者原码。

[SEAL环境与资产](seal_preflight.json)已完成249上的CPU导入和全部本地权重文件哈希。BLIP2两片权重、spiece以及作者caption-embedding权重均与Hugging Face上游LFS哈希一致。扩散模型暂用固定Manojb SD2.1镜像；stabilityai原始fp16快照同一性未认证，公开元数据请求返回401。未将此限制写成已证实模型不同。

SEAL语义入口使用官方生成、caption、SimHash和反演函数；最终布尔判定补自ICCV最终算法3和补充§9.1：1024块、每块7位、距离严格小于2.3、至少12块匹配。原入口使用所载模型的PNDM生成调度器，反演用作者DDIM代数及该调度器时间步；不擅自替换生成调度器。公开代码没有暴露论文secret salt，保留这个协议差异。

100提示词（COCO/SDP各50）开发链路已冻结于[协议](../../protocols/seal_native_20261113_v3.json)，不是论文1000图实验或独立确认。协议SHA256 `e55d8caba0c940c65e5d570fc25e1bbc56aa5207e8a1a69aa1b38c2b8426c73a`。每例保存预览、合法图、无水印图、候选caption、反演潜变量、全部1024个距离和布尔判定。无攻击、无良性扰动结果，不能据此宣称完整防御性能。

初次协调器因服务器Python不提供os.pidfd_open，在任何GPU工作进程启动前退出。原运行与[原协议](../../protocols/seal_native_20261113.json)保持冻结；v2仅改为该主机支持的Linux pidfd系统调用等待退出事件，没有轮询进度、改变模型或改变参数。

## 绑定实现与完整预算入口

[SemBind掩码检查](sembind_binding_checks.json)使用已冻结100个语义码和私有载体，sigma0、0.5、1共300次精确解绑定通过。PRC另在249生成新的私钥和6条512位消息，直接调用官方KeyGen/Encode/Decode，6条CPU往返通过。图像链路冻结于[PRC-S协议](../../protocols/sembind_prc_native_20261113_v2.json)，尚不能报告图像接受率。

[PGID-R协议](../../protocols/pgid_native_removal_20261113_v2.json)固定作者原样run.py及全部151次Imprint更新，对10个官方案例保存51/101/151次实际更新处的原判定和防御后判定。原生入口CPU导入/参数解析已通过，未改攻击函数。已有PGID-F10例结果和新旧源码可执行AST等价证据分别保留。

[队列启动收据](queue_launch.json)记录三个新协调器及协议哈希。旧等待队列在任何GPU工作进程启动前结束；新协调器增加固定3秒设备释放间隔和真实工作进程启动记录，所有数值worker、来源和样本数保持不变。执行顺序是原GS验证、SEAL100、PRC-S6、PGID-R10。旧协议留作溯源，不是重复实验结果。
