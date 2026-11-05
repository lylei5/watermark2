"""Render the confirmation100 truncated-screen report from analysis.json + posthoc.json."""
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])

a = json.loads((RUN / 'analysis.json').read_text(encoding='utf-8'))
p = json.loads((RUN / 'posthoc.json').read_text(encoding='utf-8'))
primary_label = a['primary']['label']
n = a['summary']['reference']['n']
s = a['summary']
pe = p['primary_exact']
d = p['distortion']
cs = p['cross_secret']

lines = []
add = lines.append

add('# %d 对截断轨迹筛选（confirmation100 v1.1）' % n)
add('')
add('**结论先行。** 在冻结资产的前 %d 对（哈希排序，未按结果挑选）上，按同一发布流程截断到 %s 步'
    '（发布版 151 步的 %.0f%%），普通 GS 通过 **%d/%d**（%.1f%%，Clopper–Pearson 95%% CI %.1f–%.1f%%），'
    '平均 BA **%.4f**，中位 PSNR **%.2f dB**。未处理原图对照 %d/%d 通过，平均 BA %.4f。'
    % (n, primary_label, 100 * int(primary_label) / 151, pe['k'], n, 100 * pe['rate'],
       100 * pe['clopper_pearson95'][0], 100 * pe['clopper_pearson95'][1],
       pe['mean_ba'], d['psnr_median'], p['null_control']['gs_passes'], n,
       p['null_control']['mean_ba']))
add('')
add('> 这是**截断调度**的开发筛选，不是发布版 151 步端点的复现。选择 %s 步是为了在 8 小时窗口内拿到 %d 对分布；'
    '151 步的完整 100 对确认仍需单独冻结运行。' % (primary_label, n))
add('')
add('## 1. 设计与执行')
add('')
add('- 数据：confirmation100 v1.1 冻结资产（前 %d 对：512×512 载体 + 唯一 caption，固定配对，manifest `a41bba…8220`）。' % n)
add('- 身份：每对一个独立 evaluator-only GS 身份（message/key/nonce，`private_eval`，0700/0600）；攻击端只读参考图，不读密钥。')
add('- 流程：发布版 Imprint 更新循环，Adam(lr=0.01)，损失 = 代理潜空间 MSE；本运行执行 %s 步（发布版 151）。' % primary_label)
add('- 检查点：20、%s；候选先哈希封存，再评估。' % primary_label)
add('- 评估：目标模型 50 步 DDIM 反演 + 官方 GS 解码；逐图 BA、严格阈值 %.8f；PSNR/SSIM/MS-SSIM 以载体为参照。' % 0.70703125)
add('- 资源：5 张 RTX 3090（物理 2/3/4/5/7），动态工作队列，每卡 1 进程；GPU0/1 为他人占用，GPU6 排除。')
add('')
add('## 2. 主结果')
add('')
add('| 指标 | 值 |')
add('|---|---|')
add('| 主端点 | %s 步 |' % primary_label)
add('| 普通 GS 通过 | **%d/%d**（%.1f%%） |' % (pe['k'], n, 100 * pe['rate']))
add('| 通过率 95%% CI | Clopper–Pearson %.1f–%.1f%%，Wilson %.1f–%.1f%% |'
    % (100 * pe['clopper_pearson95'][0], 100 * pe['clopper_pearson95'][1],
       100 * a['primary']['wilson95'][0], 100 * a['primary']['wilson95'][1]))
add('| 平均 BA（95%% bootstrap 区间） | %.4f（%.4f–%.4f） |'
    % (pe['mean_ba'], pe['mean_ba_bootstrap95'][0], pe['mean_ba_bootstrap95'][1]))
add('| BA 中位数（IQR） | %.4f（%.4f–%.4f） |' % (pe['median_ba'], pe['q1_ba'], pe['q3_ba']))
add('| BA 范围 | %.4f–%.4f |' % (pe['min_ba'], pe['max_ba']))
add('| PSNR 均值 / 中位 | %.2f / %.2f dB |' % (d['psnr_mean'], d['psnr_median']))
add('| PSNR（pooled MSE） | %.2f dB |' % d['pooled_psnr_db'])
add('| SSIM / MS-SSIM 均值 | %.4f / %.4f |' % (d['ssim_mean'], d['ms_ssim_mean']))
add('')
add('## 3. 轨迹与对照')
add('')
add('| 标签 | n | GS 通过 | 平均 BA | 中位 PSNR |')
add('|---|---:|---:|---:|---:|')
for label in ['reference', 'cover', '20', primary_label]:
    row = s[label]
    add('| %s | %d | %d | %.4f | %s |' % (label, row['n'], row['gs_passes'], row['mean_ba'],
                                           '—' if row['mean_psnr_db'] is None else '%.2f' % row['mean_psnr_db']))
add('')
add('- 20→%s 步平均 BA 变化：%+.4f（%.4f → %.4f）。'
    % (primary_label, p['trajectory'][primary_label]['mean_ba'] - p['trajectory']['20']['mean_ba'],
       p['trajectory']['20']['mean_ba'], p['trajectory'][primary_label]['mean_ba']))
add('- 参考图（GS 生成图、未经优化）：%d/%d 通过，平均 BA %.4f。'
    % (p['reference_control']['gs_passes'], n, p['reference_control']['mean_ba']))
add('- 未处理原图：%d/%d 通过，平均 BA %.4f（H₀ 对照）。'
    % (p['null_control']['gs_passes'], n, p['null_control']['mean_ba']))
add('- BA 与 PSNR 的 Spearman ρ = %.3f（p = %.3g，n = %d）。'
    % (p['ba_psnr_spearman']['rho'], p['ba_psnr_spearman']['p'], p['ba_psnr_spearman']['n']))
add('')
add('## 4. 密钥特异性（事后）')
add('')
add('- %d×%d 解码消息 × 秘密矩阵：对角平均 %.4f，非对角平均 %.4f，非对角最大 %.4f；'
    '非对角超过阈值的对数 **%d/%d**。'
    % (n, n, cs['diagonal_mean'], cs['offdiagonal_mean'], cs['offdiagonal_max'],
       cs['offdiagonal_ge_threshold'], cs['offdiagonal_pairs']))
add('- 该矩阵在 CPU 上补算，不改变冻结主分析。')
add('')
add('## 5. 失败案例')
add('')
add('| task | BA | PSNR |')
add('|---|---:|---:|')
for row in a['failures'][:15]:
    add('| %d | %.4f | %s |' % (row['task_id'], row['bit_accuracy'],
                                '—' if row['psnr_db'] is None else '%.2f' % row['psnr_db']))
add('')
add('失败 %d/%d 对。' % (len(a['failures']), n))
add('')
add('## 6. 与 stage31 的关系与解释边界')
add('')
add('- stage31（4 对、共享身份、151 步）：平均 BA 0.9199、4/4 通过、PSNR 23.75 dB；本运行是 %d 对、独立身份、%s 步。' % (n, primary_label))
add('- 历史轨迹（2 个任务、共享身份）：20 步 BA 0.7715，50 步 BA 0.8516；本运行 %s 步落在两者之间，方向一致。' % primary_label)
add('- 不能声称：发布版 151 步端点已复现；总体攻击成功率、FPR、ROC、SOTA；完整几何/SemBind 接受；正式人工内容评审。')
add('- 正式人工内容评审为 0；内容判断仅为助手视觉观察，不作为标签。')
add('')
add('## 7. 成本')
add('')
c = a['cost']
add('- 攻击核心时间合计 %.0f s（%.2f GPU·h），单对平均 %.0f s、中位 %.0f s。'
    % (c['attack_core_seconds_sum'], c['gpu_hours_core'], c['attack_core_seconds_mean'], c['attack_core_seconds_median']))
add('- 攻击进程墙钟合计 %.0f s。' % c['attack_process_wall_seconds_sum'])
add('')
add('## 8. 复现入口')
add('')
add('- 运行目录：`%s`' % RUN)
add('- 冻结协议：`protocols/confirmation100_{design,attacker,evaluator,controller}_v6.json`')
add('- 主分析：`analysis.json`；事后补充：`posthoc.json`（`scripts/confirmation100/posthoc100.py`）。')
add('- 私有材料：`private_eval/confirmation100_baseline_20261102T201500Z/`（不发布）。')
add('')

OUT.write_text('\n'.join(lines), encoding='utf-8')
print('wrote', OUT, 'bytes', OUT.stat().st_size)
