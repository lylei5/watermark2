# 防御基线与论文实现的差异

本项目将官方数值组件、按论文补齐的逻辑和实际执行环境分别记录。源码文件字节相同，不等于整个运行环境或模型输出与作者逐字节相同。结果与分母见[验收报告](../reports/three_baselines_20261113/report.md)，执行状态见[project_state.json](../project_state.json)。

| 方法与环节 | 当前实现 | 可主张的一致性与限制 |
|---|---|---|
| SemBind语义网络 | 官方DINOv2-Giant和已发布masker权重 | 使用已发布推理组件；没有复现训练语料和完整训练过程 |
| SemBind生成/反演 | 官方源文件、固定diffusers0.21.4 DPM调度实现；50步、inv_order0、VAE decoder优化100步 | 按最终论文实验设置；inv_order0不执行高阶固定点校正，不能把函数名理解为每条样本都达到数值精确反演 |
| SemBind内存适配 | 移动暂不使用的模块到CPU、attention slicing、decoder gradient checkpointing | 数值核心沿用作者函数，内存执行方式有适配；完整模型输出与作者环境的字节等价尚未认证 |
| SemBind符号绑定 | 论文平铺、截断、置换和符号乘法；独立私有256位HMAC置换密钥 | 完整绑定入口未发布，由论文补齐；具体密钥派生约定公开记录，秘密密钥不入Git |
| GS载体 | 原GS空间重复OTP | 区别于PGID的扁平GS；原源码166位与理想独立位167位门槛分别报告，保留历史严格182位工作点 |
| PRC载体 | SemBind发布的PRC原函数；512位消息、t3、posterior variance1.5 | Detect和Decode分别记录，同时报告其OR；不能把联合判定自动称为FPR1e-6 |
| Tree-Ring载体 | 原仓库3015283的函数，官方README实验命令channel3/ring/radius10；私有载体seed | 保留原非中心卡方p-value；载体seed受原NumPy API的32位范围限制，不宣称其具备256位密码强度；SemBind置换密钥独立 |
| GS++载体 | 尚无通过验收的完整实现 | 不能以原GS、PRC或两者任意拼接代称GS++；SemBind完整四载体矩阵未完成 |
| PGID-F/R | 固定官方dc4ad96，原样run.py和Imprint代码 | 保留原载体、生成/反演和更新次数。GS结果已验收；四载体CPU往返不是四载体图像攻击复现 |
| PGID密钥 | 原实验公开fixture | 用于重现作者实验数值；不是对未知秘密身份的安全验证 |
| SEAL公开噪声 | 原generate_initial_noise和投影seed42 | 作者已发布代码路径；没有论文中的独立secret salt |
| SEAL论文噪声 | Algorithm2的私有salt派生；SHA256、固定宽度编码及设备内私有PyTorch Generator | 按论文补齐，序列化和PRNG选择并非作者已公开约定；与公开代码分开报告 |
| SEAL生成/反演 | 所载模型PNDM生成；作者DDIM代数反演 | 请求50步的PNDM网格实际有51个时间步；不将该生成路径描述成DDIM |
| SEAL最终判定 | 每块距离严格小于2.3，至少12/1024块匹配 | 补自ICCV最终Algorithm3和补充§9.1；原CLI的未使用threshold50不是判定阈值 |
| SEAL良性处理 | 直接调用原image_transformations.py所选utils.image_distortion | Noise0.1实际是共享偏移和uint8运算；Blur8为PIL radius8；Brightness6采样范围[0,7]。报告原代码结果，并明确它们不等同于论文文字里的独立像素Gaussian噪声、8×8核和[0,6]亮度范围 |
| SEAL样本与随机性 | 固定COCO/SDP各50来源，调用set_random_seed(task) | 不是作者1000图队列。原扰动入口外层NumPy seed为img_idx，本轮helper设为task+3；记录固定样本与随机序列差异，不宣称复刻作者同一随机样本 |
| SEAL鲁棒性统计口径 | 当前七条件报告语义检测的固定12块/2.3判定 | 官方还提供无语义的逐块128码字穷举、负例99百分位阈值的TPR@1%FPR入口；不能混合为同一个防伪检测器或同一主表结果 |
| SD2.1模型 | 固定Manojb镜像0094d48，权重哈希留存 | 与stabilityai原fp16快照同一性尚未认证；这是未验证项，不是已证实不同 |

论文规格、库版本、私钥、样本量和已发表攻击预算都不得在看过结果后悄悄替换。新的参数或实现需新协议、新目录，并保留之前的结果。原论文数值、开发集正常链路和同条件防御比较是三种不同证据。
