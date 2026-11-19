"""Evidence slides with Kaiti Chinese and upright serif Latin.

The portable deck renders each text shape with the selected fonts; pictures,
connectors and videos remain separate slide objects. An editable text edition
and licensed font bundle are also exported. Preview is shared-layout rendering,
not an assertion of native PowerPoint rendering or playback validation.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from functools import lru_cache

import cv2
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.oxml.xmlchemy import OxmlElement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'report_control_20261021'))
from slide_builder import Deck as BaseDeck, sha, rgb

W,H=13.333333,7.5
SCALE=144
TEXT_SCALE=288
ROOT=Path('/mnt/sdg/datasets/real_sim_human_pipeline/pipeline_explained_20261021')
FONTS=ROOT/'fonts'
CN_NAME='AR PL UKai CN'
EN_NAME='Liberation Serif'
INK='182C3B'; MUTED='526773'; TEAL='087D7D'; BLUE='264E79'
RED='A3423D'; GOLD='9B681E'; LINE='D8E1DF'; PAPER='FAF9F5'; WHITE='FFFFFF'

def is_latin(ch):
    return ord(ch)<256 or ch in '→←↔×≥≤±'

@lru_cache(maxsize=256)
def font(size,bold=False,latin=False,scale=SCALE):
    p=FONTS/('LiberationSerif-Bold.ttf' if bold else 'LiberationSerif-Regular.ttf') if latin else FONTS/'ARPLUKaiCN.ttf'
    return ImageFont.truetype(str(p),round(size*scale/72))

def runs(text):
    buf='';last=None
    for ch in text:
        latin=is_latin(ch)
        if last is not None and latin!=last:
            yield buf,last;buf=''
        buf+=ch;last=latin
    if buf:yield buf,last

def length(text,size,bold=False,scale=SCALE):
    return sum(font(size,bold,latin,scale).getlength(s) for s,latin in runs(text))

def wrap(text,size,width,bold=False):
    lines=[]
    for para in str(text).split('\n'):
        line=''
        # Keep Latin words together when possible; CJK remains breakable.
        tokens=re.findall(r'[A-Za-z0-9_./+%≤≥=:-]+|.',para)
        for tok in tokens:
            if line and length(line+tok,size,bold)>width*SCALE:
                lines.append(line.rstrip());line=tok.lstrip()
            else:line+=tok
        lines.append(line)
    return lines

def text_sprite(e,scale):
    canvas=Image.new('RGBA',(round(e['w']*scale)+4,round(e['h']*scale)+4),(0,0,0,0))
    draw=ImageDraw.Draw(canvas)
    for li,line in enumerate(e['text'].split('\n')):
        x=0 if e['align']=='left' else (e['w']*scale-length(line,e['size'],e['bold'],scale))/2
        y=li*e['step']*scale
        for s,latin in runs(line):
            f=font(e['size'],e['bold'],latin,scale)
            # Mild synthetic weight for Chinese: no italic or sans substitution.
            sw=1 if e['bold'] and not latin and scale>=144 else 0
            draw.text((round(x)+1,round(y)+1),s,font=f,fill='#'+e['color'],anchor='lt',stroke_width=sw,stroke_fill='#'+e['color'])
            x+=f.getlength(s)
    return canvas

class Deck(BaseDeck):
    def __init__(self,out):
        super().__init__(out)
        self.font_checks=[]

    def text(self,x,y,w,h,text,size=18,color=INK,bold=False,align='left',url=None):
        lines=wrap(text,size,w,bold);step=size*1.24/72
        assert len(lines)*step<=h+.03,(self.current['title'],text,len(lines)*step,h)
        self._put(dict(kind='text',x=x,y=y,w=w,h=h,text='\n'.join(lines),size=size,color=color,bold=bold,align=align,url=url,step=step))

    def page(self,title,subtitle='',notes='',section='PIPELINE'):
        p={'title':title,'notes':notes,'elements':[]};self.pages.append(p);self.current=p
        self.rect(0,0,W,H,PAPER,PAPER)
        self.rect(.48,.36,.055,.62,TEAL,TEAL)
        self.text(.69,.32,11.95,.63,title,29,INK,True)
        if subtitle:self.text(.7,1.04,11.9,.5,subtitle,15,MUTED)
        self.line(.7,1.58,12.62,1.58,LINE,.7)
        self.text(.7,7.16,10.9,.22,f'{section}  ·  Real → Sim → Human  ·  2026.10.21  ·  证据见备注',9,MUTED)
        self.text(12.18,7.08,.5,.3,f'{len(self.pages):02d}',13,MUTED)
        return p

    def card(self,x,y,w,h,label,body='',accent=TEAL,size=18):
        self.rect(x,y,w,h,WHITE,LINE,True)
        if h<=1.4:
            lw=min(1.6,w*.36)
            self.text(x+.17,y+.16,lw,h-.28,label,size-1,accent,True)
            if body:
                bs=size-2;bw=w-lw-.48;bh=h-.28
                while len(wrap(body,bs,bw))*bs*1.24/72>bh+.02 and bs>13:bs-=1
                self.text(x+lw+.29,y+.16,bw,bh,body,bs)
        else:
            self.rect(x+.15,y+.2,.045,.3,accent,accent)
            self.text(x+.3,y+.17,w-.45,.45,label,size,accent,True)
            if body:self.text(x+.22,y+.78,w-.44,h-.94,body,size-1)

    def badge(self,x,y,w,text,color=TEAL):
        self.rect(x,y,w,.32,'EAF2F0','EAF2F0',True)
        self.text(x+.08,y+.055,w-.16,.23,text,10,color,False,'center')

    def picture(self,path,x,y,w,h,caption=None):
        super().picture(path,x,y,w,h,None)
        if caption:self.text(x,y+h+.08,w,.48,caption,13,MUTED)

    def movie(self,path,poster,x,y,w,h,caption='播放实际视频'):
        super().movie(path,poster,x,y,w,h,'')
        if caption:self.text(x,y+h+.08,w,.44,caption,13,BLUE)

    def table(self,x,y,widths,rows,row_h=.7,size=16):
        for ri,row in enumerate(rows):
            xx=x
            for v,w in zip(row,widths):
                self.rect(xx,y+ri*row_h,w,row_h,'E9F1EF' if ri==0 else WHITE,LINE)
                self.text(xx+.14,y+ri*row_h+.12,w-.28,row_h-.2,str(v),size,TEAL if ri==0 else INK,ri==0)
                xx+=w

    def label_arrow(self,x1,y1,x2,y2,label='',color=TEAL):
        self.line(x1,y1,x2,y2,color,1.3,True)
        if label:self.text(min(x1,x2),min(y1,y2)-.29,max(abs(x2-x1),.8),.25,label,11,color)

    def save(self,name='RealSimHuman_pipeline_机制与改进_楷体版'):
        prs=[Presentation(),Presentation()]
        for pr in prs:
            pr.slide_width=Inches(W);pr.slide_height=Inches(H)
            pr.core_properties.title='Real→Sim→Human：来源、机制、质量与简化设计'
        preview=self.out/'preview';preview.mkdir(exist_ok=True)
        txtdir=self.assets/'text';txtdir.mkdir(exist_ok=True)
        pngs=[]
        for pi,page in enumerate(self.pages,1):
            slides=[pr.slides.add_slide(pr.slide_layouts[6]) for pr in prs]
            canvas=Image.new('RGB',(round(W*SCALE),round(H*SCALE)),'#'+PAPER);draw=ImageDraw.Draw(canvas)
            for ei,e in enumerate(page['elements']):
                k=e['kind']
                if k=='text':
                    im=text_sprite(e,SCALE);canvas.paste(im,(round(e['x']*SCALE),round(e['y']*SCALE)),im)
                    high=text_sprite(e,TEXT_SCALE);tp=txtdir/f'{pi:02d}_{ei:03d}.png';high.save(tp)
                    sh=slides[0].shapes.add_picture(str(tp),Inches(e['x']),Inches(e['y']),Inches(high.width/TEXT_SCALE),Inches(high.height/TEXT_SCALE))
                    sh.name='Text_'+e['text'].replace('\n',' ')[:60]
                    sh._element.nvPicPr.cNvPr.set('descr',e['text'])
                    if e.get('url'):sh.click_action.hyperlink.address=e['url']
                    sh=slides[1].shapes.add_textbox(*[Inches(e[z]) for z in ('x','y','w','h')])
                    tf=sh.text_frame;tf.clear();tf.auto_size=MSO_AUTO_SIZE.NONE;tf.word_wrap=False
                    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                    for li,line in enumerate(e['text'].split('\n')):
                        p=tf.paragraphs[0] if li==0 else tf.add_paragraph();p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=Pt(e['step']*72)
                        if e['align']=='center':p.alignment=2
                        for s,latin in runs(line):
                            r=p.add_run();r.text=s;r.font.name=EN_NAME if latin else CN_NAME;r.font.size=Pt(e['size']);r.font.bold=e['bold'];r.font.italic=False;r.font.color.rgb=rgb(e['color'])
                            for tag in ['a:ea','a:cs']:
                                node=OxmlElement(tag);node.set('typeface',CN_NAME);r._r.get_or_add_rPr().append(node)
                            if e.get('url'):r.hyperlink.address=e['url']
                elif k=='rect':
                    for slide in slides:
                        typ=MSO_SHAPE.ROUNDED_RECTANGLE if e['radius'] else MSO_SHAPE.RECTANGLE
                        sh=slide.shapes.add_shape(typ,*[Inches(e[z]) for z in ('x','y','w','h')]);sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['fill']);sh.line.color.rgb=rgb(e['line']);sh.line.width=Pt(.7)
                        if e['radius']:sh.adjustments[0]=.07
                    box=[round(e['x']*SCALE),round(e['y']*SCALE),round((e['x']+e['w'])*SCALE),round((e['y']+e['h'])*SCALE)]
                    if e['radius']:draw.rounded_rectangle(box,radius=12,fill='#'+e['fill'],outline='#'+e['line'],width=1)
                    else:draw.rectangle(box,fill='#'+e['fill'],outline='#'+e['line'],width=1)
                elif k=='line':
                    for slide in slides:
                        sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,*[Inches(e[z]) for z in ('x1','y1','x2','y2')]);sh.line.color.rgb=rgb(e['color']);sh.line.width=Pt(e['width'])
                        if e['arrow']:
                            node=OxmlElement('a:tailEnd');node.set('type','triangle');sh.line._get_or_add_ln().append(node)
                    pts=[e[k]*SCALE for k in ['x1','y1','x2','y2']];draw.line(pts,fill='#'+e['color'],width=max(1,round(e['width']*SCALE/72)))
                    if e['arrow']:
                        a=math.atan2(pts[3]-pts[1],pts[2]-pts[0]);l=11
                        draw.polygon([(pts[2],pts[3]),(pts[2]-l*math.cos(a-.5),pts[3]-l*math.sin(a-.5)),(pts[2]-l*math.cos(a+.5),pts[3]-l*math.sin(a+.5))],fill='#'+e['color'])
                elif k in ['image','movie']:
                    path=e['path'] if k=='image' else e['poster']
                    for slide in slides:
                        if k=='image':slide.shapes.add_picture(path,*[Inches(e[z]) for z in ('x','y','w','h')])
                        else:slide.shapes.add_movie(e['path'],*[Inches(e[z]) for z in ('x','y','w','h')],poster_frame_image=path,mime_type='video/mp4')
                    im=Image.open(path).convert('RGB').resize((round(e['w']*SCALE),round(e['h']*SCALE)),Image.Resampling.LANCZOS)
                    canvas.paste(im,(round(e['x']*SCALE),round(e['y']*SCALE)))
            for slide in slides:slide.notes_slide.notes_text_frame.text=page['notes']
            canvas.save(preview/f'slide_{pi:02d}.png');pngs.append(canvas)
        out=self.out/(name+'.pptx');prs[0].save(out)
        editable=self.out/(name+'_可编辑.pptx');prs[1].save(editable)
        pngs[0].save(self.out/(name+'_预览.pdf'),save_all=True,append_images=pngs[1:],resolution=SCALE)
        sheet=Image.new('RGB',(3*640,math.ceil(len(pngs)/3)*360),'#'+LINE)
        for i,im in enumerate(pngs):sheet.paste(im.resize((640,360)),((i%3)*640,(i//3)*360))
        sheet.save(preview/'contact_sheet.jpg',quality=93)
        (self.out/'layout.json').write_text(json.dumps(self.pages,ensure_ascii=False,indent=2)+'\n')
        (self.out/'media_bindings.json').write_text(json.dumps(self.bindings,ensure_ascii=False,indent=2)+'\n')
        (self.out/'render_method.json').write_text(json.dumps({'slides':len(self.pages),'portable_text':'Kaiti and upright serif rendered text shapes with source text in accessibility descriptions','editable_text':True,'native_PowerPoint_render':False,'preview_method':'shared_layout_Pillow','embedded_videos':sum(e['kind']=='movie' for p in self.pages for e in p['elements'])},indent=2)+'\n')
        with zipfile.ZipFile(self.out/(name+'_编辑与字体.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            z.write(editable,editable.name)
            for fname in ['ARPLUKaiCN.ttf','LiberationSerif-Regular.ttf','LiberationSerif-Bold.ttf','ARPHIC-LICENSE.txt','LIBERATION-LICENSE.txt','font_manifest.json']:z.write(FONTS/fname,'fonts/'+fname)
            z.writestr('使用说明.txt','演示版文本已按楷体与端正衬线英文字体绘制，无需安装字体。\n可编辑版请先安装 fonts 目录字体；字体许可证随包附带。\n所有预览来自同一布局，不代表已用 PowerPoint 原生播放验证。\n')
        return out
