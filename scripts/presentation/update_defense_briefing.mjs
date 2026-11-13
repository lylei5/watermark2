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
edit(1,7,'现有方法、研究进展与实例绑定思路',28,green,true,'center');
edit(1,8,'汇报人：李艳磊    2026 年 11 月 13 日',24,green,true,'center');
edit(2,7,'研究目标\n正常编辑后保留身份\n换图和篡改后识别冒用',22,green,true,'center');
[[11,'先明确要防什么'],[12,'水印去除、身份伪造与内容篡改'],[16,'三篇论文分别怎么做'],[17,'防御机制、对比方法与测试攻击'],[21,'我们做到哪一步'],[22,'基线实验、问题诊断与改进空间'],[26,'准备怎样改进'],[27,'从语义相似转向具体发布实例']].forEach(([id,v])=>edit(2,id,v,[12,17,22,27].includes(id)?18:26,[12,17,22,27].includes(id)?dark:green,true));

function panel(n,title,body,x,y,w,h,size=23) {
  slide(n).shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'#F0F5F1',line:{fill:'#C1D8C9',width:1}});
  text(n,title,x+16,y+12,w-32,36,25,green,true);
  text(n,body,x+16,y+58,w-32,h-70,size);
}
for(let n=3;n<=10;n++) await clearBody(n);

header(3,'01','先明确防什么：去除、伪造和篡改','防伪是主线；正常图像经过允许编辑后，仍应保留水印身份');
panel(3,'水印去除：让原图查不出来','原本带水印的图像，处理后检测失败。\n\n例：Imprint Removal、再生成去除。',36,124,588,230,26);
panel(3,'身份伪造：让别的图冒用身份','另一张图通过目标水印的检测。\n\n例：Imprint Forgery、Reprompt。',657,124,588,230,26);
text(3,'几个攻击名称，分别在做什么',48,385,1182,40,27,green,true);
text(3,'Imprint：通过扰动改变反演后的水印判定，包含伪造和去除两个方向。\nReprompt：复用估计出的水印噪声，换提示词生成不同内容。\nCat Attack：在已有图像中贴入对象，改变内容但尝试保留原水印。',48,438,1182,132,25);
text(3,'防御目标：正常编辑后仍认得原图，内容换了以后能识别身份不匹配。',48,614,1182,62,27,green,true);

for(const n of [4,5,6]) {
  const c=content.slides[n-1];
  header(n,'02',c.title,c.subtitle);
  const xs=[36,455,874];
  c.steps.forEach((a,i)=>panel(n,a[0],a[1],xs[i],132,371,168,23));
  text(n,'→',413,190,36,42,28,gold,true,'center');
  text(n,'→',832,190,36,42,28,gold,true,'center');
  text(n,c.mechanism,48,316,1182,76,26,green,true);
  panel(n,'论文和谁比较',c.comparators,36,412,588,235,23);
  panel(n,'论文测试了什么',c.attacks,657,412,588,235,22);
  text(n,c.reference,48,670,1182,30,16,dark);
}

header(7,'03','三条防御路线，留下哪些改进空间','从论文的设计取舍看问题，再用我们的实验定位具体环节');
table(7,[['方法','解决问题的方式','进一步要解决的问题'],['SemBind','语义码调制已有水印\n换内容后让解绑失配','同提示词一致性较强\n语义相近的不同实例仍需区分'],['PGID','修复受扰动的反演过程\n兼顾伪造拒绝与去除恢复','参数随载体和场景变化\n还要保住正常编辑后的检出'],['SEAL','描述绑定分块噪声\n结合空间检查发现局部编辑','重新描述可能产生漂移\n几何变化会影响分块对应']],36,128,1209,326,[200,470,539],24);
text(7,'共同问题：系统要知道“这是不是同一张发布图”',48,484,1182,48,28,green,true);
text(7,'同一张图的正常编辑应当保留身份；同语义的另一张图应当有自己的身份。\n改进方向：让内容表示围绕具体实例稳定下来，再与水印身份绑定。',48,551,1182,112,27);

header(8,'03','当前进展：基线已跑通，问题已定位','已完成核心链路与诊断实验，下一步聚焦实例表征');
table(8,[['已经做了什么','实验带来的认识'],['三项基线核心流程\n生成、反演与检测','已经有可运行的比较基础\n完整横向评测按统一条件继续推进'],['PGID 完整 Imprint 测试\n以及正常图的常规处理','防伪和恢复有效\n正常处理后也出现合法图误拒'],['SEAL 参考漂移诊断\nSemBind 语义码距离分析','描述重提取会影响参考稳定性\n辅助图漂移与不同实例差异存在重叠']],36,128,1209,333,[450,759],24);
text(8,'方法探索给出的方向',48,487,1182,40,27,green,true);
text(8,'载体噪声变体与生成轨迹参考的尝试，尚未带来稳定收益。\n下一步保留成熟载体，先解决“同图编辑稳定、不同实例分开”。',48,546,1182,108,27);
text(8,'实现范围与实验记录见逐页备注和仓库报告。',48,675,1182,24,16,dark);

header(9,'04','改进方案：围绕最终发布图绑定实例身份','方案设计：先确定发布图，再建立实例参考；原生水印载体保持不变');
text(9,'训练目标：同图编辑保持一致，不同实例拉开距离',48,124,1182,42,28,green,true);
text(9,'同图的 JPEG、缩放等允许编辑作为正例；同提示词的其他实例作为负例。\n让正常变化留在恢复范围内，让内容替换落在范围外。',48,185,1182,88,26);
table(9,[['环节','拟采用的机制'],['生成','用原生 GS / PRC 生成并确定最终发布图'],['登记','从发布图提取实例码，建立经认证的密钥恢复记录'],['验证','候选图经内容匹配与容错恢复，取回对应载体密钥'],['检测','使用恢复的密钥，执行原生水印检测']],36,302,1209,245,[205,1004],24);
text(9,'核心思路：由“同一类内容”推进到“同一张发布图”的身份绑定。',48,577,1182,48,27,green,true);
text(9,'主要对照：同样实例码的直接距离门控，比较容错密钥恢复带来的收益。',48,644,1182,35,22);

header(10,'04','下一步：把实例区分变成防御收益','论文主线：同图编辑保持身份，不同实例分开身份');
panel(10,'先验证实例表征','训练鲁棒实例码。\n\n比较同图编辑与不同实例的距离，找到稳定的恢复范围。',36,137,371,310,25);
panel(10,'再验证绑定机制','比较直接内容门控与模糊提取器等容错恢复方案。\n\n分析防御收益、记录大小与检测成本。',455,137,371,310,25);
panel(10,'最后做防御比较','与三项基线同条件比较。\n\n沿用已发表攻击，在相同正常图保留水平下比较防伪。',874,137,371,310,25);
text(10,'创新准备落在两个相连的问题上',48,492,1182,42,28,green,true);
text(10,'怎样学到对正常编辑稳定、对不同实例敏感的内容码？\n怎样让这种实例区分可靠地控制水印身份恢复？',48,555,1182,111,29,green,true);


for(let i=0;i<10;i++) {
  const c=content.slides[i];
  slide(i+1).speakerNotes.text = c.notes+'\n\n导师可能追问\n'+(c.questions??[]).map(x=>x.q+'\n'+x.a).join('\n\n')+'\n\n证据来源\n'+c.sources.join('\n');
}
await fs.writeFile(path.join(buildDir,'content-inspection.json'),JSON.stringify(await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']}),null,2));
await (await PresentationFile.exportPptx(p)).save(path.join(buildDir,'candidate.pptx'));
await fs.writeFile(path.join(buildDir,'build.json'),JSON.stringify({source,source_sha256:createHash('sha256').update(await fs.readFile(source)).digest('hex'),slide_count:10,content:contentPath},null,2));
console.log('Wrote candidate.pptx, 10 editable slides');
