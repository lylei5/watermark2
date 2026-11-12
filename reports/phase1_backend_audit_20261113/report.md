# PRC组件与模型资产核验

249上的PRC原码完成 **8/8 检测与512位消息精确解码**。测试仅涉及16384维载体采样、可逆符号绑定及原译码器，不是PRC-S图像实验，也不是实测FPR保证。

源码固定为作者[PRC-Watermark](https://github.com/XuandongZhao/PRC-Watermark/tree/3f372c1d9cafcd0732331c0d00f0430ee822a389)，提交`3f372c1d9cafcd0732331c0d00f0430ee822a389`。服务器已有副本的所用文件与Git对象逐字节一致；PRC核心与SemBind发布版本的AST一致。使用16384位码字、512位消息、t=3、名义FPR=1e-6、后验方差1.5。密钥和载体留在服务器。

依赖使用作者指定的Galois0.4.1、LDPC0.1.51、NumPy1.26.0，置于独立目录`envs/phase1_prc_extra`。PyTorch2.5.1和SciPy1.13.1仍与作者旧环境不同；8次无图像往返不证明所有数值路径等价。已有Galois0.4.10/LDPC2.4.1组合在第一次完整Decode出现浮点到GF(2)类型错误，见[失败记录](initial_failure.json)及[初始协议](initial_protocol.json)。没有修改第三方源码。

| 资产 | 核验结果 | 范围 |
|---|---|---|
| SD1.5 | 本机249已有完整扩散组件，UNet权重SHA256与对应LFS对象名一致；关键配置与权重已实际读盘校验 | 尚未运行该代理的攻击 |
| CLIP ViT-L/14 | 本机249已有配置、分词器和1.71GB模型文件，哈希已记录 | 可用于补充质量评估；不是论文ViT-g/14规格的等价物 |
| PRC | 原码8/8往返成功，原检测与Decode均通过 | 尚未完成PRC-S图像基线 |
| GS++ | 未在所核查作者材料及本项目资产中找到可信完整实现 | 不自行拼装并冠以原生GS++名称；不填写实测成绩 |

PRC的原`decode.py`将Detect与Decode作OR组合。若分别采用名义1e-6参数，不能仅凭各自阈值就宣称组合判决的FPR也为1e-6；后续需明确采用的完整判定规则及其null。

[完整结果与资产哈希](complete.json)；[冻结协议](protocol.json)；[执行代码](../../scripts/defense/audit_phase1_backends.py)。服务器运行目录为`runs/phase1_backend_audit_20261113_v2`，初次兼容性失败保留于无v2后缀的目录。
