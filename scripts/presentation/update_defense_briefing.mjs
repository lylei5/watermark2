// Update the supplied SYSU deck with audited research evidence.
// Run with the bundled @oai/artifact-tool on NODE_PATH/module search path.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const [source, buildDir, contentPath] = process.argv.slice(2);
if (!source || !buildDir || !contentPath) throw new Error('Usage: source.pptx build-dir content.json');
await fs.mkdir(buildDir, {recursive:true});
const content = JSON.parse(await fs.readFile(contentPath, 'utf8'));
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const green = '#005A27', gold = '#B78600', dark = '#303030', red = '#B92F28';
const font = '微软雅黑';
const slide = n => p.slides.getItem(n-1);
function edit(n, id, value, size, color, bold, align) {
  const shape = slide(n).shapes.items.find(x=>x.id===String(id));
  if (!shape) throw new Error(`Missing source shape ${n}/${id}`);
  shape.text = value;
  if (size) shape.text.style = {typeface:font,fontSize:size,color:color??dark,bold:bold??false,alignment:align??'left',verticalAlignment:'middle',wrap:true};
  return shape;
}
function text(n, value, x,y,w,h, size=23, color=dark, bold=false, align='left') {
  const s=slide(n).shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=value;
  s.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',wrap:true};
  s.text.insets=0;
  return s;
}
async function clearBody(n) {
  const snap=await p.inspect({select:`$.slides[${n-1}].elements[*]`,include:['id','kind','position']});
  const ids=snap.records.filter(r=>{
    const f=p.resolve(r.id);
    return r.kind==='table'||r.kind==='chart'||(r.kind==='image'&&!['2','3','5'].includes(f.id))||(['shape','textbox'].includes(r.kind)&&!['4','6','7'].includes(f.id));
  }).map(r=>r.id);
  p.delete(ids);
}
function table(n, values, x,y,w,h,widths,size=21) {
  const t=slide(n).tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  const all=t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length});
  all.assign({textStyle:{typeface:font,fontSize:size,color:dark},margins:{left:12,right:10,top:8,bottom:8},anchor:'center',borders:{top:{fill:'#FFFFFF',width:1},bottom:{fill:'#FFFFFF',width:1},left:{fill:'#FFFFFF',width:1},right:{fill:'#FFFFFF',width:1}}});
  for(let r=0;r<values.length;r++) {
    t.rows[r].height=h/values.length;
    const row=t.cells.block({row:r,column:0,rowCount:1,columnCount:values[0].length});
    row.assign({fill:r===0?green:r%2?'#F6F8F5':'#E8F2EB'});
    if(r===0) row.assign({textStyle:{typeface:font,fontSize:size,bold:true,color:'#FFFFFF'}});
  }
  return t;
}
function header(n, section, title, subtitle) {
  edit(n,4,section,28,'#FFFFFF',true,'center');
  edit(n,6,title,32,green,true);
  edit(n,7,subtitle,18,gold,true);
}

edit(1,6,'扩散模型语义水印防御',44,green,true,'center');
edit(1,7,'基线验证与实例绑定研究',30,green,true,'center');
edit(1,8,'汇报人：李艳磊    2026 年 11 月 13 日',24,green,true,'center');

edit(2,7,'研究目标\n建立可复核的防御基线\n解决发布实例的可靠内容绑定',22,green,true,'center');
[[11,'研究问题与 SemBind 机制'],[12,'水印来源认证与图像内容绑定'],[16,'三项防御基线的验证进展'],[17,'SemBind、PGID、SEAL 的已完成结果'],[21,'失败实验给出的机制约束'],[22,'合法内容漂移与不同实例的接受区域重叠'],[26,'实例绑定防御与验证计划'],[27,'内容密钥的可行条件及论文贡献边界']].forEach(([id,v])=>edit(2,id,v,[12,17,22,27].includes(id)?18:26,[12,17,22,27].includes(id)?dark:green,true));

header(3,'01','研究目标：可靠识别发布图像及其合法编辑','研究对象是防御；评测只使用已发表攻击的原始实现与完整预算');
edit(3,10,'水印载体负责检测，内容绑定负责关联实例',24,'#FFFFFF',true);
edit(3,11,'Gaussian Shading、Tree-Ring、PRC 将水印写入生成噪声，再通过反演和原生检测器验证。\n\n载体匹配本身未必能区分：同一发布图的压缩版本，以及内容不同的水印伪造图。\n\n防御目标：保留合法编辑，同时拒绝冒用目标来源记录的其他实例。',23,dark);
edit(3,14,'论文需要回答的核心问题',24,'#FFFFFF',true);
edit(3,15,'在相同合法检出率下，内容绑定能否进一步降低伪造接受？\n\n同时报告图像质量、计算成本，以及密钥和记录的访问权限。',24,dark);
edit(3,17,'合法保留',34,green,true,'center');
edit(3,18,'干净图及允许的良性编辑',19,dark,true,'center');
edit(3,20,'伪造拒绝',34,red,true,'center');
edit(3,21,'Imprint / Reprompt',18,dark,true,'center');
edit(3,23,'当前重点：完成 SemBind、PGID、SEAL 的实现核验，并识别内容绑定真正需要解决的机制问题。\n可发表的贡献需要独立、同条件实验支撑，不能仅依赖语义距离变大或小样本零接受。',23,green,true);

header(4,'01','SemBind 的语义绑定与当前实现','ICML 2026；官方发布组件，未公开的完整绑定入口按论文补齐');
edit(4,10,'阶段一：由辅助图提取 1024 位语义码',24,'#FFFFFF',true);
edit(4,12,'① DPM 生成辅助图',22,green,true);
edit(4,13,'同一提示词下生成辅助图，50 步，CFG 7.5',20,dark);
edit(4,15,'② DINOv2-giant',22,green,true);
edit(4,16,'读取全局 CLS 特征，维度为 1536',20,dark);
edit(4,18,'③ 官方语义网络',22,green,true);
edit(4,19,'加载发布权重，得到 1024 维输出',20,dark);
edit(4,21,'④ 二值语义码',22,green,true);
edit(4,22,'按符号量化，训练包含同提示词及良性增强一致性',20,dark);
edit(4,25,'阶段二：语义码扩展、私有置换与载体噪声符号调制',24,'#FFFFFF',true);
edit(4,27,'论文中的统一绑定方式',23,green,true);
edit(4,28,'语义码平铺到 16,384 坐标，按比例截断后秘密置换。\n将符号掩码乘到载体初始噪声，再执行 50 步 DPM 生成。',20,dark);
edit(4,30,'绑定比例 σ',23,gold,true);
edit(4,31,'GS / TR：σ = 1\nPRC / GS++：σ = 0.5',22,dark);
edit(4,34,'阶段三：候选图自身语义解绑，再交给原生载体检测器',24,'#FFFFFF',true);
edit(4,35,'反演使用官方 DPM 入口：50 步、空提示词 CFG 3、VAE 解码器优化 100 步。\n候选图重新提码并生成掩码，解绑后使用 GS 投票、TR p-value 或 PRC 检测与解码。\n私有 HMAC 置换派生由项目明确约定；按论文补齐不等于获得作者完整实验代码。',22,dark);

await clearBody(5);
header(5,'02','三项防御基线：已完成的图像验证','各方法保留原生载体与工作点；下表展示验证范围，不作横向性能排名');
table(5,[['链路','合法图接受','无水印接受','当前证据'],['SemBind GS-S','100 / 100','0 / 100','原 GS 空间 OTP；完整 DPM 链路'],['SemBind PRC-S','6 / 6','0 / 6','512 位载荷全部精确解码'],['SemBind TR-S','6 / 6','0 / 6','原 Tree-Ring 函数与 p-value'],['SEAL 公开代码','83 / 100','0 / 100','原生成、语义描述与反演'],['SEAL 论文 salt 实现','91 / 100','0 / 100','补齐私有 salt；单独报告'],['PGID-F / PGID-R','各 10 / 10','F：0 / 10','官方 GS，完整 151 次攻击更新']],36,112,1209,364,[295,180,185,549],22);
text(5,'复现可信度',48,493,300,38,25,green,true);
text(5,'语义组件：100 图 × 10 条件，与官方入口逐位一致，最大差异为 0。\n源码、协议和逐图产物均留存哈希；DPM 严格确定性检查在所测 3 例中逐值一致。',48,535,1182,70,22);
text(5,'边界：GS++、其他 PGID 图像载体及完整论文规模尚未验收。0/100 不能证明 FPR = 10⁻⁶。',48,625,1182,40,20,red,true);

await clearBody(6);
header(6,'02','PGID：完整预算下的伪造拒绝与去除恢复','作者官方 GS 案例各 10 个；保留原代码及三个保存检查点');
table(6,[['实际更新次数','伪造：原检测接受','伪造：PGID-F 接受','去除：原检测接受','去除：PGID-R 接受'],['51','10 / 10','0 / 10','0 / 10','10 / 10'],['101','10 / 10','0 / 10','0 / 10','10 / 10'],['151','10 / 10','0 / 10','0 / 10','10 / 10']],36,120,1209,248,[185,256,256,256,256],22);
text(6,'正常图对照',48,390,520,40,26,green,true);
text(6,'PGID-F 和 PGID-R 各保留 10/10 张正常水印图。\nPGID-F 对未加水印封面接受 0/10。',48,438,1182,70,24);
text(6,'这批实验验证了官方防御在完整 151 次更新下的作用。\n三个检查点重复使用相同来源，不能合并成 30 个独立样本。',48,535,1182,74,24,green,true);
text(6,'范围：公开 fixture 身份与开发样本；尚不支持未知秘密身份安全或跨方法优越性结论。',48,631,1182,36,19,red,true);

await clearBody(7);
header(7,'02','SEAL：语义参考漂移影响正常检出与编辑鲁棒性','两个实现各 100 个来源、7 个条件，共 1,400 条判定；分母包含干净图漏检');
table(7,[['条件','公开代码','论文 salt 实现'],['Clean','83 / 100','91 / 100'],['Rot75','0 / 100','0 / 100'],['JPEG25','44 / 100','48 / 100'],['CropScale75','2 / 100','0 / 100'],['Blur8','0 / 100','0 / 100'],['Noise0.1¹','77 / 100','73 / 100'],['Brightness6¹','49 / 100','53 / 100']],36,118,650,428,[260,195,195],22);
text(7,'误差分解支持的观察',720,124,515,42,26,green,true);
text(7,'干净图漏检：公开路径 17 张，论文路径 9 张。\n\n诊断时使用生成时噪声作为参考，两组均可通过 100/100。\n\n本批漏检与从候选图重新提取语义参考有关。生成时参考仅用于诊断。',720,183,515,304,23);
text(7,'统计口径',48,565,180,36,23,green,true);
text(7,'本表采用候选自身描述及固定 12 块 / 2.3 语义规则。\n官方另有无语义穷举的 TPR@1%FPR 入口，两者不能与论文 Figure 6 混作同一结果。',48,605,1182,62,20);
text(7,'¹ 保留原函数实际定义；Noise 为共享偏移，Brightness 范围为 [0,7]，Blur8 为 PIL radius 8。',48,679,1182,24,15,dark);

await clearBody(8);
header(8,'03','失败实验指出：容忍内容漂移，也会接纳不同实例','100 提示词开发诊断，COCO / SDP 各 50；以下半径在同一批数据上选择');
table(8,[['数据集','汉明半径 / 1024','辅助图—最终图在半径内','同提示词不同 seed 也在半径内'],['COCO','123','48 / 50','41 / 50'],['SDP','228','48 / 50','45 / 50'],['合并','179','95 / 100','88 / 100']],36,116,1209,225,[185,225,379,420],22);
text(8,'增加纠错半径不能自行产生实例区分。88/100 是码分布重叠，不是攻击接受率。',48,358,1182,40,24,red,true);
table(8,[['已测试方向','观察结果','对方法设计的约束'],['减少载体改动','候选合法 44/60，对照 59/60','潜变量更接近，不保证内容绑定更稳定'],['条件高斯传输','候选 55/60，重排对照 57/60','没有建立超出简单对照的收益'],['生成轨迹参考','第 35 步 P5：0.905 / 0.892','未达到预注册的两个数据集均 >0.95']],36,420,1209,204,[240,400,569],21);
text(8,'前两项来自 6 来源项目适配器的开发实验，各含 10 条件；均未推进攻击扩展。当前尚未验证新防御优势。',48,645,1182,38,18,dark);

await clearBody(9);
header(9,'04','拟研究主线：最终发布实例的内容绑定','核心问题是恢复区域的分离与生成时序；模糊提取器是候选协议工具');
text(9,'机制目标',48,123,220,40,27,green,true);
text(9,'同一发布图的允许编辑可恢复目标身份；其他实例应在恢复区域之外。\n保留原生 GS / PRC 载体，验证内容绑定是否提供额外防伪收益。',48,170,1182,80,26);
table(9,[['设计环节','需要解决的问题'],['实例码器','同一发布图的良性编辑作正例；同提示词不同 seed 作实例负例'],['内容参考与生成时序','辅助图派生载体密钥会改变最终实例，需要先证明可恢复性'],['候选可部署路线','原生随机密钥生成后，以最终图指纹建立经认证的密钥封装记录'],['验证规则','内容恢复、距离检查、密钥确认及原生载体检测同时通过']],36,278,1209,255,[285,924],22);
text(9,'贡献检验',48,552,240,38,25,green,true);
text(9,'必须优于“原生载体检测 + 同一实例指纹的距离门控”，并报告外部记录与检索成本。\nFE 不自动保证半径外拒绝或 FPR = 10⁻⁶；辅助数据认证、内容熵和多记录误报仍需分析。',48,598,1182,74,22,red,true);

header(10,'04','下一步：以独立同条件实验决定是否形成论文','阶段性结论：基线核心链路已有证据，新防御仍处于机制设计与可行性验证阶段');
edit(10,10,'一、完善基线与工作点',24,'#FFFFFF',true);
edit(10,11,'补齐 SemBind / SEAL 的既有攻击评测及未覆盖载体。\n\n统一数据来源、检测工作点、允许编辑和图像质量口径。\n\n开发集、校准集、确认集按来源与提示词族隔离。',23,dark);
edit(10,14,'二、验证实例分离',24,'#FFFFFF',true);
edit(10,15,'先测试最终发布图的鲁棒实例指纹，冻结允许的编辑范围。\n\n对照原语义码、实例训练码器与直接距离门控。\n\n检查合法编辑尾部与不同实例分布能否分开，再决定是否加入 FE。',23,dark);
edit(10,18,'三、确认论文贡献',24,'#FFFFFF',true);
edit(10,19,'仅用已发表 Imprint、Reprompt 等原始基线与完整预算。\n\n在匹配绝对合法检出率下，比较伪造接受和去除恢复。\n\n同时报告画质、耗时、存储，以及密钥与辅助数据的安全边界。',23,dark);
edit(10,21,'拟检验的论文命题：通过约束“同一发布实例”的恢复区域，提高内容绑定的防伪能力。\n成立条件：独立数据上优于强基线和简单门控，且合法编辑性能、画质及成本可以接受。',24,green,true);

for(let i=0;i<10;i++) slide(i+1).speakerNotes.text = content.slides[i].notes+'\n\n证据来源\n'+content.slides[i].sources.join('\n');
await fs.writeFile(path.join(buildDir,'content-inspection.json'),JSON.stringify(await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']}),null,2));
await (await PresentationFile.exportPptx(p)).save(path.join(buildDir,'candidate.pptx'));
await fs.writeFile(path.join(buildDir,'build.json'),JSON.stringify({source,source_sha256:createHash('sha256').update(await fs.readFile(source)).digest('hex'),slide_count:10,content:contentPath},null,2));
console.log('Wrote candidate.pptx, 10 editable slides');
