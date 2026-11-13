"""Create the per-slide Chinese speaker guide from the shared deck content."""
import argparse
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

def set_font(style, size, bold=False):
    style.font.name='微软雅黑'
    style.font.size=Pt(size)
    style.font.bold=bold
    style.font.color.rgb=RGBColor(0,0,0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'微软雅黑')

def link(p, label, url):
    rel=p.part.relate_to(url,RT.HYPERLINK,is_external=True)
    h=OxmlElement('w:hyperlink');h.set(qn('r:id'),rel)
    r=OxmlElement('w:r');pr=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'365F91');pr.append(color)
    font=OxmlElement('w:rFonts');font.set(qn('w:eastAsia'),'微软雅黑');pr.append(font)
    size=OxmlElement('w:sz');size.set(qn('w:val'),'18');pr.append(size)
    r.append(pr);t=OxmlElement('w:t');t.text=label;r.append(t);h.append(r);p._p.append(h)

def main():
    a=argparse.ArgumentParser();a.add_argument('content',type=Path);a.add_argument('output',type=Path);args=a.parse_args()
    c=json.loads(args.content.read_text(encoding='utf8'))
    d=Document();s=d.sections[0]
    s.page_width=Cm(21);s.page_height=Cm(29.7)
    s.top_margin=Cm(1.9);s.bottom_margin=Cm(1.8);s.left_margin=Cm(2.0);s.right_margin=Cm(2.0)
    s.header_distance=Cm(.8);s.footer_distance=Cm(.8)
    for name,size,bold in [('Normal',11.5,False),('Title',22,True),('Heading 1',17,True),('Heading 2',12,True),('Caption',9,False)]:
        set_font(d.styles[name],size,bold)
    for border in d.styles['Title'].element.findall('.//' + qn('w:pBdr')):
        border.getparent().remove(border)
    pf=d.styles['Normal'].paragraph_format;pf.line_spacing=1.22;pf.space_after=Pt(6)
    for name in ['Heading 1','Heading 2']:
        f=d.styles[name].paragraph_format;f.space_before=Pt(9);f.space_after=Pt(5);f.keep_with_next=True
    d.styles['Heading 1'].paragraph_format.page_break_before=True
    d.core_properties.title=c['guide']['title'];d.core_properties.subject='逐页口头讲稿与导师问答'
    d.core_properties.author='李艳磊'
    d.add_paragraph(c['guide']['title'],'Title')
    d.add_paragraph('李艳磊　2026 年 11 月 13 日')
    d.add_paragraph(c['guide']['intro'])
    d.add_paragraph('开场用这段话','Heading 2');d.add_paragraph(c['guide']['opening'])
    for key,title in [('progress','现在做到了什么'),('innovation','创新准备落在哪里'),('decision','下一步先拿什么结果')]:
        d.add_paragraph(title,'Heading 2');d.add_paragraph(c['guide'][key])
    d.add_paragraph('三个方法，一句话记住','Heading 2')
    for t in ['SemBind：用语义码调制已有水印，主要防 Imprint 与 Reprompt 伪造。',
              'PGID：在检测端修复反演过程，主要防 Imprint 伪造和去除。',
              'SEAL：用描述构造分块噪声，检查噪声复用伪造和局部内容篡改。']:
        d.add_paragraph(t)
    for i,x in enumerate(c['slides'],1):
        d.add_paragraph(f'第 {i} 页  {x["title"]}','Heading 1')
        p=d.add_paragraph();p.add_run(x['point']).bold=True
        d.add_paragraph('口头讲稿','Heading 2')
        for para in x['notes'].split('\n'): d.add_paragraph(para)
        d.add_paragraph('导师可能追问','Heading 2')
        for qa in x['questions']:
            p=d.add_paragraph();p.paragraph_format.space_before=Pt(5);p.paragraph_format.keep_with_next=True
            p.add_run(qa['q']).bold=True
            d.add_paragraph(qa['a'])
        p=d.add_paragraph(style='Caption');p.add_run('证据索引　')
        for j,url in enumerate(x['sources'],1):
            if j>1:p.add_run('　')
            link(p,f'来源 {j}',url)
    p=s.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field)
    args.output.parent.mkdir(parents=True,exist_ok=True);d.save(args.output)
    md=['# '+c['guide']['title'],'',c['guide']['intro'],'']
    for key,title in [('opening','开场'),('progress','当前进展'),('innovation','创新思路'),('decision','下一步')]:
        md.extend(['## '+title,'',c['guide'][key],''])
    for i,x in enumerate(c['slides'],1):
        md.extend([f'## 第 {i} 页 {x["title"]}','',x['point'],'','### 口头讲稿','',x['notes'],'','### 导师可能追问',''])
        for qa in x['questions']:md.extend(['**'+qa['q']+'**','',qa['a'],''])
        md.extend(['证据：'+'，'.join(f'[来源 {j}]({url})' for j,url in enumerate(x['sources'],1)),''])
    args.output.with_suffix('.md').write_bytes(('\n'.join(md).rstrip()+'\n').encode('utf8'))
    print(args.output)

if __name__=='__main__':main()
