"""Render the audited bounded GS mechanism evidence into a Chinese research report."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main(run):
    result = read(run/'results.json')
    plan = read(run/'protocol.json')
    complete = read(run/'analysis_complete.json')
    assert complete['status']=='complete'
    gpu = read(run/'gpu_complete.json')
    names = [('p01','1%'),('p03','3%'),('p06','6%')]
    labels = dict(clean='未处理原图',random_crop_ratio='保留区域0.6',random_drop_ratio='抹除区域0.8',
        brightness_factor='亮度参数6',jpeg_ratio='JPEG25',gaussian_blur_r='高斯模糊4',
        gaussian_std='作者高斯噪声0.05',median_blur_k='中值滤波7',sp_prob='椒盐噪声0.05',resize_ratio='缩小0.25再还原')
    passed = [label for name,label in names if result['ratios'][name]['passed']]
    lead = ('结论：'+ '、'.join(passed)+'通过这次小规模机制筛查，尚未验证攻击下的防御能力。') if passed else '结论：三个写入比例均未达到通过标准，这条不动点写入实现停止，不继续扩大实验或运行攻击。'
    lines = ['# 256位GS随图指纹机制验证','',lead,'',
        '50个提示词全部完成，每个比例最多三轮。判据是256位逐位完全一致，不把相似度过线当成收敛。未收敛样本保留在所有联合通过率的分母中。','',
        '| 写入比例 | 三轮内逐位收敛 | 同时满足漂移限制 | 原图联合通过 | 九类处理联合通过 | 达标 |',
        '|---|---:|---:|---:|---:|---|']
    for name,label in names:
        r=result['ratios'][name]
        lines.append(f"| {label} | {r['converged']}/50 | {r['strict_converged']}/50 | {r['conditions']['clean']['joint']}/50 | {r['nine_edits']['joint']}/450（{r['nine_edits']['joint']/4.5:.1f}%） | {'是' if r['passed'] else '否'} |")
    lines += ['', '目标分别为：收敛且漂移达标至少45/50；原图联合至少45/50；九类处理联合至少405/450；每类GS检测数均不下降。联合通过指“GS检测通过，且反演读回的码与查询图指纹匹配”。', '',
        '## 稳定性：噪声改动后还有多少位在变','',
        '| 比例 | 第1/2/3轮累计收敛 | 首轮翻转位数均值 | 停止轮翻转位数均值 | 停止轮非对称距离均值 | 相对基准图SSCD距离均值 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in names:
        r=result['ratios'][name]; d=r['mechanism_diagnostics']; by=r['convergence_by_round']
        lines.append(f"| {label} | {by['1']}/{by['2']}/{by['3']} | {d['first_round_flips']['mean']:.2f} | {r['final_flips']['mean']:.2f} | {r['final_drift']['mean']:.4f} | {r['baseline_sscd_distance']['mean']:.4f} |")
    lines += ['',f"匹配分数阈值固定为{plan['similarity_threshold']:.10f}；合法距离阈值的三分之一为{plan['drift_limit']:.10f}。二值码与连续查询比较即使是自身也有量化距离，所以上表的非对称距离不等于纯粹新增的图像变化；连续SSCD距离和位翻转数同时保留。",'',
        '## 读回与联合合法率','',
        '| 处理 | 原生GS | 1% GS | 3% GS | 6% GS | 1% 联合 | 3% 联合 | 6% 联合 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in result['conditions']:
        fields=[f"{result['baseline'][c]['gs_accept']}/50"]
        fields += [f"{result['ratios'][name]['conditions'][c]['gs']}/50" for name,_ in names]
        fields += [f"{result['ratios'][name]['conditions'][c]['joint']}/50" for name,_ in names]
        lines.append('| '+labels[c]+' | '+' | '.join(fields)+' |')
    lines += ['', '| 处理 | 1% 平均误码率 | 3% 平均误码率 | 6% 平均误码率 |', '|---|---:|---:|---:|']
    for c in result['conditions']:
        values=[f"{100*result['ratios'][name]['conditions'][c]['ber']['mean']:.2f}%" for name,_ in names]
        lines.append('| '+labels[c]+' | '+' | '.join(values)+' |')
    lines += ['', '误码率以实际写入的256位明文码为真值。它和“生成后指纹已经改变了多少”是两个独立问题。全部条件使用同一提示词的相同处理种子。常规处理保持已有GS作者函数的原始行为，包括亮度随机采样和高斯噪声uint8运算；没有改成另一种同名操作。', '',
        '## 画面变化','', '| 比例 | PSNR均值（dB） | LPIPS均值 | LPIPS中位数 | LPIPS最大值 |', '|---|---:|---:|---:|---:|']
    for name,label in names:
        q=result['ratios'][name]['quality']
        lines.append(f"| {label} | {q['psnr']['mean']:.2f} | {q['lpips']['mean']:.4f} | {q['lpips']['median']:.4f} | {q['lpips']['max']:.4f} |")
    lines += ['', '这些指标衡量与同提示词、同初始噪声原生GS图的差异，不是绝对画质评价，也不是FID。', '',
        '![预先选定的六个提示词对照](figures/preselected_comparison.png)', '',
        '图中每行从左到右为原生GS、1%、3%、6%。展示任务在看结果前固定为1、2、6、52、53、55；原图不调色，统一显示尺寸。另提供[LPIPS变化最大的三个提示词](figures/largest_lpips_comparison.png)，这张补充图按结果事后选择。','',
        '## 样本量与不确定性','',
        '| 比例 | 收敛率及Wilson95%区间 | 原图联合率及Wilson95%区间 | 九类联合率及提示词bootstrap95%区间 |',
        '|---|---|---|---|']
    def interval(p,ci):
        return f'{100*p:.1f}% [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]'
    for name,label in names:
        r=result['ratios'][name]; c=r['conditions']['clean']; e=r['nine_edits']
        lines.append(f"| {label} | {interval(r['converged']/50,r['converged_wilson95'])} | {interval(c['joint']/50,c['joint_wilson95'])} | {interval(e['joint']/450,e['joint_cluster95'])} |")
    lines += ['', '九类处理共450次查询，但只有50个独立提示词；区间按提示词成组重采样5000次。这里是既有拆分上的机制筛查，不是独立确认或普遍安全性保证。','',
        '## 资源与复核','',
        f"只使用249的GPU3；累计GPU分配时间上界{result['gpu']['cumulative_gpu_hours_upper_bound']:.4f}小时，包含两次启动失败约13.57秒。完成{gpu['generation_calls']}次生成、{gpu['queries']}次反演，每次完整50步，记录{gpu['unet_calls']}次UNet调用。无攻击。SEAL的GPU7/5/4未被迁移或终止。LPIPS、统计、图表和独立复核均在GPU退出后使用249 CPU。",'',
        f"独立复核重新计算了{result['audit']['gs_replays']}条GS判定、{result['audit']['payload_replays']}次读码、{result['audit']['score_replays']}个指纹分数，以及{result['audit']['round_replays']}轮的指纹变化。判定差异为0。详细记录见[results.json](results.json)和[independent_audit.json](independent_audit.json)。",'',
        '## 实现范围与结论边界','',
        '三个写入比例分别占163、491、983个噪声位置。1%只有163个位置，单靠正负号装不下256位，因此使用每位置两位的高斯四区间；3%/6%使用符号位及循环重复。它们不是仅改变比例的纯消融。使用固定GS密钥派生的异或掩码，不加认证和额外纠错；验证端只用供应方密钥、图中读回的码、查询图特征和公开投影，不使用逐图登记的参考。保存的完整参考和明文码仅用于实验误码/漂移审计。', '',
        '50个提示词来自既有评估拆分，COCO/SDP各25个。3个超长提示词由原管线按77-token上限截断，各比例一致。模型SD2.1，生成CFG7.5、反演空提示词CFG1，均为50步DDIM；未使用PGID防御或VAE优化。GS固定历史阈值BA>181/256，未宣称已经完成统一自然图FPR校准。', '',
        '此次结论只针对这套冻结实现和预算。不会据此启动新变体，也没有验证任何攻击；统一SemBind/PGID/SEAL对比仍是主线。', '',
        f"协议SHA256：`{result['protocol_sha256']}`。服务器资产：`{run}`。代码：[执行脚本](../../scripts/defense/gs_fingerprint_fixedpoint.py)、[独立分析](../../scripts/defense/analyze_gs_fingerprint_fixedpoint.py)、[协议说明](../../docs/GS_FINGERPRINT_FIXEDPOINT_PROTOCOL.md)。",'']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(run/'report.md'),'passed':result['passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    main(p.parse_args().run)
