"""Local editable PPTX + same-layout PNG/PDF previews; no model calls.

The preview is rendered from the shared layout, not claimed to be a PowerPoint
or LibreOffice render. Media originals are never changed. All displayed frame
extractions and embedded videos retain their source paths and SHA256 bindings.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt
from pptx.oxml.xmlchemy import OxmlElement

W, H = 13.333333, 7.5
SCALE = 144
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_NAME = 'Noto Sans CJK SC'
INK, GRAY, BLUE, RED, GREEN = '17202D', '586578', '2457A7', 'B44142', '167A68'
LIGHT, WHITE = 'DCE2E8', 'FFFFFF'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def rgb(s):
    return RGBColor.from_string(s)


def font(size, bold=False):
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH,
                              round(size * SCALE / 72), index=2)


def wrap(text, size, width, bold=False):
    """Explicit CJK line breaks keep editable PowerPoint text predictable."""
    f = font(size, bold)
    out = []
    for paragraph in str(text).split('\n'):
        line = ''
        for ch in paragraph:
            if line and f.getlength(line + ch) > width * SCALE:
                out.append(line.rstrip())
                line = ch.lstrip()
            else:
                line += ch
        out.append(line)
    return out


class Deck:
    def __init__(self, output_dir):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.assets = self.out / 'assets'
        self.assets.mkdir(exist_ok=True)
        self.pages = []
        self.bindings = []
        self.current = None

    def page(self, title, subtitle='', notes=''):
        p = {'title': title, 'notes': notes, 'elements': []}
        self.pages.append(p)
        self.current = p
        self.text(.55, .28, 12.25, .54, title, 25, bold=True)
        if subtitle:
            self.text(.57, .87, 12.15, .47, subtitle, 12, GRAY)
        self.line(.56, 1.36, 12.77, 1.36, LIGHT)
        self.text(.57, 7.17, 11.7, .18,
                  'Real → Sim → 合成人类参考  |  实际证据 · 2026-10-21  |  详细来源见备注',
                  8, GRAY)
        self.text(12.3, 7.10, .5, .3, str(len(self.pages)).zfill(2), 11, GRAY)
        return p

    def _put(self, e):
        if e['kind'] not in {'line'}:
            x,y,w,h = [e[k] for k in ('x','y','w','h')]
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert x+w <= W+.002 and y+h <= H+.002, (self.current['title'],e)
        self.current['elements'].append(e)

    def text(self, x,y,w,h,text,size=16,color=INK,bold=False,align='left',url=None):
        lines = wrap(text, size, w, bold)
        step = size * 1.26 / 72
        assert len(lines)*step <= h + .025, (self.current['title'], text, len(lines)*step, h)
        self._put(dict(kind='text', x=x,y=y,w=w,h=h,text='\n'.join(lines),
                       size=size,color=color,bold=bold,align=align,url=url,step=step))

    def rect(self,x,y,w,h,fill=WHITE,line=LIGHT,radius=False):
        self._put(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,line=line,radius=radius))

    def line(self,x1,y1,x2,y2,color=LIGHT,width=1,arrow=False):
        self._put(dict(kind='line',x1=x1,y1=y1,x2=x2,y2=y2,color=color,width=width,arrow=arrow))

    def picture(self,path,x,y,w,h,caption=None):
        path = Path(path).resolve()
        assert path.exists(), path
        im=Image.open(path)
        scale=min(w/im.width,h/im.height)
        dw,dh=im.width*scale,im.height*scale
        self._put(dict(kind='image',path=str(path),x=x+(w-dw)/2,y=y+(h-dh)/2,w=dw,h=dh))
        self.bindings.append({'type':'image','path':str(path),'sha256':sha(path)})
        if caption:
            self.text(x,y+h+.06,w,.43,caption,11,GRAY)

    def frame(self,path,index,label=''):
        path=Path(path).resolve()
        cap=cv2.VideoCapture(str(path))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        index=count-1 if index == -1 else index
        assert 0 <= index < count,(path,index,count)
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read();cap.release()
        assert ok,(path,index)
        target=self.assets/(sha(path)[:12]+f'_frame_{index:06d}.png')
        if not target.exists(): cv2.imwrite(str(target),frame)
        self.bindings.append({'type':'decoded_frame','source':str(path),'source_sha256':sha(path),
                              'frame_index':index,'fps':fps,'frame_count':count,'path':str(target),
                              'label':label,'sha256':sha(target)})
        return target

    def crop(self,path,box):
        """Literal source-pixel crop for presentation only, not a detector label."""
        path=Path(path).resolve();im=Image.open(path)
        l,t,r,b=box
        assert 0 <= l < r <= im.width and 0 <= t < b <= im.height
        target=self.assets/(sha(path)[:12]+'_crop_'+'_'.join(map(str,box))+'.png')
        if not target.exists():im.crop(box).save(target)
        self.bindings.append({'type':'presentation_crop','source':str(path),'source_sha256':sha(path),
                              'literal_source_pixel_box':box,'path':str(target),'sha256':sha(target),
                              'used_as_acceptance_measurement':False})
        return target

    def movie(self,path,poster,x,y,w,h,caption='点击播放实际输出'):
        path=Path(path).resolve();poster=Path(poster).resolve()
        im=Image.open(poster);scale=min(w/im.width,h/im.height)
        dw,dh=im.width*scale,im.height*scale
        self._put(dict(kind='movie',path=str(path),poster=str(poster),x=x+(w-dw)/2,y=y+(h-dh)/2,w=dw,h=dh))
        self.bindings.append({'type':'embedded_video','path':str(path),'sha256':sha(path)})
        if caption:self.text(x,y+h+.04,w,.4,caption,10,BLUE)

    def bullet(self,x,y,w,label,body,color=BLUE,size=16):
        self.text(x,y,w,.36,label,size,color,bold=True)
        self.text(x,y+.43,w,.82,body,size-1)

    def table(self,x,y,widths,rows,row_h=.69,size=14):
        for ri,row in enumerate(rows):
            xx=x
            for ci,(v,w) in enumerate(zip(row,widths)):
                self.rect(xx,y+ri*row_h,w,row_h,fill='EFF3F7' if ri==0 else WHITE)
                self.text(xx+.12,y+ri*row_h+.12,w-.24,row_h-.16,str(v),size,
                          BLUE if ri==0 else INK,bold=ri==0)
                xx+=w

    def save(self,name='RealSimHuman_实际案例与控制进展_20261021'):
        prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
        prs.core_properties.title='Real→Sim→合成人类参考：实际案例、耗时费用与控制进展'
        prs.core_properties.subject='实际来源绑定与阶段性资格；不宣称未验证的全链成功'
        preview=self.out/'preview';preview.mkdir(exist_ok=True)
        pngs=[]
        for pi,page in enumerate(self.pages,1):
            slide=prs.slides.add_slide(prs.slide_layouts[6])
            canvas=Image.new('RGB',(round(W*SCALE),round(H*SCALE)),'white')
            draw=ImageDraw.Draw(canvas)
            for e in page['elements']:
                k=e['kind']
                if k=='text':
                    sh=slide.shapes.add_textbox(*[Inches(e[z]) for z in ('x','y','w','h')])
                    sh.name='Text_'+e['text'].split('\n')[0][:50]
                    tf=sh.text_frame;tf.clear();tf.auto_size=MSO_AUTO_SIZE.NONE;tf.word_wrap=False
                    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                    for li,txt in enumerate(e['text'].split('\n')):
                        p=tf.paragraphs[0] if li==0 else tf.add_paragraph()
                        p.text=txt;p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=Pt(e['step']*72)
                        if e['align']=='center':p.alignment=2
                        for r in p.runs:
                            r.font.name=FONT_NAME;r.font.size=Pt(e['size']);r.font.bold=e['bold'];r.font.color.rgb=rgb(e['color'])
                            if e.get('url'):r.hyperlink.address=e['url']
                        f=font(e['size'],e['bold']);xx=e['x']*SCALE
                        if e['align']=='center':xx+=(e['w']*SCALE-f.getlength(txt))/2
                        draw.text((xx,(e['y']+li*e['step'])*SCALE),txt,font=f,fill='#'+e['color'],anchor='lt')
                elif k=='rect':
                    typ=MSO_SHAPE.ROUNDED_RECTANGLE if e['radius'] else MSO_SHAPE.RECTANGLE
                    sh=slide.shapes.add_shape(typ,*[Inches(e[z]) for z in ('x','y','w','h')])
                    sh.fill.solid();sh.fill.fore_color.rgb=rgb(e['fill']);sh.line.color.rgb=rgb(e['line']);sh.line.width=Pt(.7)
                    box=[e['x']*SCALE,e['y']*SCALE,(e['x']+e['w'])*SCALE,(e['y']+e['h'])*SCALE]
                    draw.rectangle(box,fill='#'+e['fill'],outline='#'+e['line'],width=1)
                elif k=='line':
                    sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,*[Inches(e[z]) for z in ('x1','y1','x2','y2')])
                    sh.line.color.rgb=rgb(e['color']);sh.line.width=Pt(e['width'])
                    points=[e['x1']*SCALE,e['y1']*SCALE,e['x2']*SCALE,e['y2']*SCALE]
                    draw.line(points,fill='#'+e['color'],width=max(1,round(e['width']*SCALE/72)))
                    if e['arrow']:
                        tail=OxmlElement('a:tailEnd');tail.set('type','triangle');sh.line._get_or_add_ln().append(tail)
                        a=math.atan2(points[3]-points[1],points[2]-points[0]);l=9
                        draw.polygon([(points[2],points[3]),(points[2]-l*math.cos(a-.45),points[3]-l*math.sin(a-.45)),(points[2]-l*math.cos(a+.45),points[3]-l*math.sin(a+.45))],fill='#'+e['color'])
                elif k in {'image','movie'}:
                    p=e['path'] if k=='image' else e['poster']
                    if k=='image':slide.shapes.add_picture(p,*[Inches(e[z]) for z in ('x','y','w','h')])
                    else:slide.shapes.add_movie(e['path'],*[Inches(e[z]) for z in ('x','y','w','h')],poster_frame_image=p,mime_type='video/mp4')
                    im=Image.open(p).convert('RGB').resize((round(e['w']*SCALE),round(e['h']*SCALE)),Image.Resampling.LANCZOS)
                    canvas.paste(im,(round(e['x']*SCALE),round(e['y']*SCALE)))
            slide.notes_slide.notes_text_frame.text=page['notes']
            target=preview/f'slide_{pi:02d}.png';canvas.save(target);pngs.append(canvas)
        output=self.out/(name+'.pptx');prs.save(output)
        pngs[0].save(self.out/(name+'_预览.pdf'),save_all=True,append_images=pngs[1:],resolution=SCALE)
        thumb_w=640;thumb_h=360;columns=3;rows=math.ceil(len(pngs)/columns)
        sheet=Image.new('RGB',(columns*thumb_w,rows*thumb_h),'#dce2e8')
        for i,im in enumerate(pngs):sheet.paste(im.resize((thumb_w,thumb_h)),((i%columns)*thumb_w,(i//columns)*thumb_h))
        sheet.save(preview/'contact_sheet.jpg',quality=92)
        (self.out/'layout.json').write_text(json.dumps(self.pages,ensure_ascii=False,indent=2)+'\n')
        (self.out/'media_bindings.json').write_text(json.dumps(self.bindings,ensure_ascii=False,indent=2)+'\n')
        (self.out/'render_method.json').write_text(json.dumps({'method':'shared_layout_Pillow_preview','native_PowerPoint_render':False,'editable_pptx':True,'embedded_videos':sum(e['kind']=='movie' for p in self.pages for e in p['elements']),'slides':len(self.pages)},indent=2)+'\n')
        return output
