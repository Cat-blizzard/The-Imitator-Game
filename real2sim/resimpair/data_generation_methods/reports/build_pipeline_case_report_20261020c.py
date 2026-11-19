#!/usr/bin/python3
"""Build an evidence-led Chinese PDF from actual pipeline artifacts.

Run with /usr/bin/python3 (ReportLab and OpenCV are system packages).
No model API, generation, training, or quality release occurs here.
The optional --case-manifest adds this turn's bounded repair experiment.
All displayed video frames are decoded from the stated local media.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

WORK = Path("/mnt/sdf/dataset/zhouxunzhe/zhiqi")
ROOT = Path("/mnt/sdg/datasets/real_sim_human_pipeline")
BATCH = ROOT / "batch_20261020b"
OUT = ROOT / "state_conditioned_20261020c" / "report"
JOBS = BATCH / "jobs"
W, H = landscape(A4)
M = 37
CW = W - 2 * M
FONT = "ReportSong"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe(text):
    return html.escape(str(text)).replace("\n", "<br/>")


class Report:
    def __init__(self, output, font_path=None):
        global FONT
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.assets = self.output.parent / "assets"
        self.assets.mkdir(exist_ok=True)
        self.media = self.output.parent / "videos"
        self.media.mkdir(exist_ok=True)
        if font_path and Path(font_path).exists():
            pdfmetrics.registerFont(TTFont(FONT, str(font_path)))
        else:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            FONT = "STSong-Light"
        self.c = canvas.Canvas(str(self.output), pagesize=(W, H), pageCompression=1)
        self.c.setTitle("任务关键状态约束的 Real→Sim→Synthetic-Human 数据管线")
        self.c.setAuthor("PolicyGen 数据管线实验记录")
        self.page_no = 0
        self.page_records = []
        self.frame_records = []
        self.video_records = []
        self.y = H - M
        self.styles = {
            "body": ParagraphStyle("body", fontName=FONT, fontSize=12, leading=19,
                                   wordWrap="CJK", spaceAfter=8, alignment=TA_LEFT),
            "small": ParagraphStyle("small", fontName=FONT, fontSize=9.5, leading=14.5,
                                    wordWrap="CJK", spaceAfter=4),
            "cell": ParagraphStyle("cell", fontName=FONT, fontSize=10.2, leading=15.5,
                                   wordWrap="CJK", spaceAfter=1),
            "title": ParagraphStyle("title", fontName=FONT, fontSize=21, leading=29,
                                    wordWrap="CJK", spaceAfter=8),
            "sub": ParagraphStyle("sub", fontName=FONT, fontSize=15, leading=23,
                                  wordWrap="CJK", spaceAfter=7),
        }

    def page(self, title, subtitle=""):
        if self.page_no:
            self._finish_page()
            self.c.showPage()
        self.page_no += 1
        self.page_records.append({"page": self.page_no, "title": title, "min_y": H})
        self.c.bookmarkPage(f"page_{self.page_no}")
        self.c.addOutlineEntry(title, f"page_{self.page_no}", level=0, closed=False)
        self.y = H - 35
        self.p(title, style="title")
        if subtitle:
            self.p(subtitle, style="small")
        self.c.setStrokeColor(colors.HexColor("#444444"))
        self.c.line(M, self.y - 1, W - M, self.y - 1)
        self.y -= 15

    def _finish_page(self):
        self.c.setFont(FONT, 8.5)
        self.c.setFillColor(colors.HexColor("#555555"))
        self.c.drawString(M, 19, "2026-10-20  |  实际文件与证据记录  |  未启动策略训练")
        self.c.drawRightString(W - M, 19, f"{self.page_no:02d}")
        self.c.setFillColor(colors.black)

    def p(self, text, style="body", x=None, width=None, gap=None):
        x = M if x is None else x
        width = CW if width is None else width
        para = Paragraph(safe(text), self.styles[style])
        _, height = para.wrap(width, H)
        if self.y - height < 39:
            raise ValueError(f"Page {self.page_no} overflows: y={self.y}, h={height}, text={text[:80]}")
        para.drawOn(self.c, x, self.y - height)
        self.y -= height + (self.styles[style].spaceAfter if gap is None else gap)
        self.page_records[-1]["min_y"] = min(self.page_records[-1]["min_y"], self.y)

    def bullets(self, lines, style="body"):
        for line in lines:
            self.p("• " + line, style=style)

    def table(self, rows, widths=None, size=10.2):
        widths = widths or [CW / len(rows[0])] * len(rows[0])
        style = ParagraphStyle("t", parent=self.styles["cell"], fontSize=size, leading=size * 1.5)
        vals = [[Paragraph(safe(c), style) for c in row] for row in rows]
        t = Table(vals, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.65, colors.black),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#AAAAAA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        _, h = t.wrap(CW, H)
        if self.y - h < 39:
            raise ValueError(f"Table page {self.page_no} overflows: {h} at {self.y}")
        t.drawOn(self.c, M, self.y - h)
        self.y -= h + 12
        self.page_records[-1]["min_y"] = min(self.page_records[-1]["min_y"], self.y)

    def gap(self, n=8):
        self.y -= n

    def image_row(self, items, height=160, captions=True):
        """Items contain path,label; optional crop is literal source pixel box."""
        n = len(items)
        gutter = 12
        width = (CW - (n - 1) * gutter) / n
        top = self.y
        max_caption = 0
        for i, item in enumerate(items):
            path = Path(item["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            x = M + i * (width + gutter)
            im = Image.open(path).convert("RGB")
            if item.get("crop"):
                im = im.crop(item["crop"])
                target = self.assets / (sha(path)[:12] + "_crop_" + str(i) + ".jpg")
                im.save(target, quality=94)
                path = target
            iw, ih = im.size
            scale = min(width / iw, height / ih)
            dw, dh = iw * scale, ih * scale
            self.c.drawImage(str(path), x + (width - dw) / 2, top - dh, dw, dh, mask="auto")
            self.c.setStrokeColor(colors.HexColor("#BBBBBB"))
            self.c.rect(x + (width - dw) / 2, top - dh, dw, dh, stroke=1, fill=0)
            if captions:
                p = Paragraph(safe(item["label"]), self.styles["small"])
                _, ch = p.wrap(width, H)
                p.drawOn(self.c, x, top - height - ch - 5)
                max_caption = max(max_caption, ch)
        self.y = top - height - max_caption - 14
        if self.y < 39:
            raise ValueError(f"Image row overflows on page {self.page_no}: {self.y}")
        self.page_records[-1]["min_y"] = min(self.page_records[-1]["min_y"], self.y)

    def video_frame(self, video, idx, label=""):
        video = Path(video)
        key = hashlib.sha256(f"{video.resolve()}:{idx}".encode()).hexdigest()[:16]
        out = self.assets / f"frame_{key}_{idx:06d}.jpg"
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        idx = count - 1 if idx == -1 else idx
        if not 0 <= idx < count:
            raise ValueError((str(video), idx, count))
        if not out.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {video}:{idx}")
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cap.release()
        self.frame_records.append({
            "video": str(video), "frame_index": idx, "fps": fps,
            "timestamp_seconds": idx / fps, "image": str(out),
            "timestamp_basis": "verified_CFR_index_over_fps",
        })
        return {"path": str(out), "label": label or f"第{idx}帧 / {idx/fps:.3f}秒"}

    def register_video(self, name, source):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        target = self.media / name
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            try:
                os.link(source.resolve(), target)
            except OSError:
                shutil.copy2(source.resolve(), target)
        if sha(target) != sha(source):
            raise ValueError(f"Report video attachment conflicts: {target}")
        rel = "videos/" + name
        item = {"name": name, "relative_link": rel, "source": str(source.resolve()), "sha256": sha(source)}
        if item not in self.video_records:
            self.video_records.append(item)
        return rel

    def link_video(self, name, source, label=None):
        rel = self.register_video(name, source)
        text = label or name
        self.c.setFont(FONT, 9.5)
        self.c.drawString(M, self.y - 12, "视频附件：" + text)
        self.c.linkURL(rel, (M, self.y - 15, M + min(CW, 480), self.y + 1), relative=1, thickness=0)
        self.y -= 23
        if self.y < 39:
            raise ValueError(f"Video link overflows on page {self.page_no}")
        self.page_records[-1]["min_y"] = min(self.page_records[-1]["min_y"], self.y)

    def web_link(self, label, url):
        self.c.setFont(FONT, 9.3)
        self.c.drawString(M, self.y - 11, label)
        self.c.linkURL(url, (M, self.y - 14, M + min(CW, 540), self.y + 1), relative=0, thickness=0)
        self.y -= 22

    def diagram(self, boxes, edges, height=300):
        top = self.y
        for i, box in enumerate(boxes):
            x, y, width, bh = box["rect"]
            x += M
            y = top - y - bh
            self.c.setStrokeColor(colors.black)
            self.c.setLineWidth(0.7)
            self.c.roundRect(x, y, width, bh, 3, stroke=1, fill=0)
            p = Paragraph(safe(box["text"]), self.styles["cell"])
            _, ph = p.wrap(width - 12, bh - 5)
            p.drawOn(self.c, x + 6, y + (bh - ph) / 2)
        for start, end in edges:
            self.c.line(M + start[0], top - start[1], M + end[0], top - end[1])
            ex, ey = M + end[0], top - end[1]
            sx, sy = M + start[0], top - start[1]
            import math
            ang = math.atan2(ey - sy, ex - sx)
            for da in (-.5, .5):
                self.c.line(ex, ey, ex - 7 * math.cos(ang + da), ey - 7 * math.sin(ang + da))
        self.y -= height + 10

    def save(self, provenance):
        self._finish_page()
        self.c.save()
        out = self.output.parent
        (out / "report_provenance.json").write_text(json.dumps({
            "pdf": str(self.output), "pdf_sha256": sha(self.output),
            "pages": self.page_records, "displayed_frames": self.frame_records,
            "videos": self.video_records, "inputs": provenance,
            "training_started": False, "source_images_are_actual_decoded_frames": True,
        }, ensure_ascii=False, indent=2))
        lines = ["# PDF 视频附件", "", "PDF不内嵌播放视频；相对链接指向本目录 videos/。附件为普通文件（同盘可使用硬链接）；复制整个报告目录即可保留相对路径。附件包含角色别名和交叉复用，不按文件数计算独立样本。阅读器若限制相对视频链接，可直接在videos目录打开同名文件。", ""]
        for v in self.video_records:
            lines += [f"- [{v['name']}]({v['relative_link']})", f"  - 原文件：`{v['source']}`", f"  - SHA256：`{v['sha256']}`"]
        (out / "VIDEO_INDEX.md").write_text("\n".join(lines))


def build(args):
    manifest = load(BATCH / "input_manifest.json")
    jobs = {j["job_id"]: j for j in manifest["jobs"]}
    report_data = load(BATCH / "BATCH_REPORT.json")
    new = load(args.case_manifest) if args.case_manifest else None
    r = Report(args.output, args.font)
    black = "apple_bowl_transfer_891910000"
    white = "apple_bowl_transfer_891911000"
    book = "closed_book_to_open_box_891912000"
    cab = "right_cabinet_open_only_891914000"
    bundle = {x["job_id"]: Path(x["directory"]) for x in load(BATCH / "review_bundle/bundle_manifest.json")["cases"]}
    generated = lambda key: JOBS / key / "generated_human.mp4"
    real = lambda key: bundle[key] / "01_real_human_task_window.mp4"
    donor = lambda key: bundle[key] / "02_sim_generation_donor.mp4"
    query = lambda key: bundle[key] / "04_independent_robot_query.mp4"
    for key in (black, white, book, cab):
        for label, fn in (("real_window", real), ("sim_donor", donor), ("generated", generated), ("robot_query", query)):
            r.register_video(f"{key}_{label}.mp4", fn(key))

    r.page("任务关键状态约束的数据生成管线", "Real → Sim → Synthetic Human  |  方法、实际案例、失败诊断与训练使用边界")
    r.p("要解决的问题", "sub")
    r.p("从一个真实人类任务出发，生成可批量变化的机器人执行，再生成与任务对应的人类参考。目标是保存对象角色、必要动作与结果，使参考视频能够提供机器人观察本身不包含的任务信息。")
    r.p("本报告优先回答：哪些问题需要视觉条件，哪些需要局部观测，哪些必须由程序约束；哪些模块已经运行，哪些仍是待验证的设计。")
    r.table([
        ["已完成基础批次", "实际数量／状态", "正确解释"],
        ["真实任务来源 / 任务类型", "3个窗口 / 3类", "两个苹果布局共用同一真人来源"],
        ["合成人类视频", "4段：3段新生成＋1段复用", "不等于4条正式训练样本"],
        ["显式配对 / CPU读取", "4条 / 12次，读取无拒绝", "图像与动作接口通过；质量另行审核"],
        ["自动完整证据通过 / 训练", "0 / 未启动", "覆盖不足与真实失败分开记录"],
    ], [180, 225, CW - 405])
    if new:
        r.p("本轮新增：" + new.get("summary", "关键状态条件与缺陷修复实验，详见本报告新增案例部分。"), style="small")
    r.p("范围：本报告展示真实文件，不报告策略成功率、真机泛化或已获得人类3D动作真值。", "small")

    r.page("01  当前管线：任务规格贯穿生成和验收", "流程框表示数据与证据；不把仿真真值直接继承给生成的人类视频")
    boxes = [
        {"rect": (0, 0, 230, 63), "text": "真人任务窗口\n原视频、帧窗、初态、关键步骤、终态"},
        {"rect": (267, 0, 230, 63), "text": "任务规格与已有仿真执行\n对象角色、原生动作、阶段、原速视频"},
        {"rect": (534, 0, 230, 63), "text": "关键状态条件准备\n初态／接触／隐藏物体显露／终态"},
        {"rect": (534, 104, 230, 63), "text": "人类关键图编辑＋前置检查\n对象、几何、手数量、可见性"},
        {"rect": (267, 104, 230, 63), "text": "能力匹配的视频生成\n实际支持的首图／尾图／参考条件"},
        {"rect": (0, 104, 230, 63), "text": "解码与局部审核\n实际PTS、任务事件、对象连续性"},
        {"rect": (0, 208, 230, 63), "text": "分级与有界纠错\n缺证据→补帧；内容错→修复／停止"},
        {"rect": (267, 208, 230, 63), "text": "明确参考→机器人query配对\n来源、独立执行、split、动作归属"},
        {"rect": (534, 208, 230, 63), "text": "用途分级与CPU读取\n任务参考资格≠物理GT≠训练批准"},
    ]
    edges = [((230,31),(267,31)),((497,31),(534,31)),((649,63),(649,104)),
             ((534,135),(497,135)),((267,135),(230,135)),((115,167),(115,208)),
             ((230,239),(267,239)),((497,239),(534,239))]
    r.diagram(boxes, edges, height=283)
    r.p("原生仿真端：保留可计算的动作、相机和任务状态。生成端：将这些状态转为模型实际接收的视觉条件。审核端：重新测量生成结果，绝不把输入要求当成输出已经满足的证据。", "body")
    r.p("图中候选可以进入inspection做读取诊断；这不等于训练准入。补帧回到审核，修关键图回到生成前检查，超预算或未实现分支停止。上批仅首图＋文字，本轮实际分支见新案例。", "small")

    r.page("02  初始输入：三个真实任务窗口", "来自EgoDex；起始帧包含，结束帧不包含；示范允许与仿真布局和本体不同")
    r.image_row([
        r.video_frame(real(black), 90, "苹果换碗：真实片段246帧，30Hz，8.20秒"),
        r.video_frame(real(book), 35, "闭合书入开盒：原片[190,261)，71帧，2.367秒"),
        r.video_frame(real(cab), 40, "只开右柜门：原子片段[0,72)，2.40秒"),
    ], height=185)
    r.table([
        ["任务", "固定含义", "不能顺带包含的后续动作"],
        ["苹果换碗", "从占用碗抓起苹果，放进原来空的碗并撤手", "不能换成倾倒碗、移动碗或换另一颗苹果"],
        ["闭合书入开盒", "闭合书从桌面进入开着的盒子，手释放", "完整整理桌面视频里的后续物体不能混入"],
        ["开右柜门", "抓右把手、打开右门、释放；保留内部物体", "后面的取物、动左门、关门不属于此窗口"],
    ], [100, 340, CW - 440], size=10)
    r.p("书本片段在报告视频附件中已准确裁出；柜门输入本来就是原子裁片，不再次叠加父视频时间偏移。", "small")
    r.link_video(f"{black}_real_window.mp4", real(black), "真实苹果换碗完整窗口")
    r.link_video(f"{book}_real_window.mp4", real(book), "真实书本入盒精确窗口")
    r.link_video(f"{cab}_real_window.mp4", real(cab), "真实只开右柜门原子窗口")

    r.page("03  任务规格：从文字描述变成可检查的条件")
    r.table([
        ["要素", "以苹果换碗为例", "程序或模型如何使用"],
        ["对象角色", "apple；source_bowl；target_bowl；distractor", "角色绑定后固定，不能随新图左右位置重新猜目标"],
        ["初始关系", "apple inside source；target empty", "初图核对，失败则不进入视频生成"],
        ["必要事件", "grasp → lift → transfer → release → withdraw", "用于选择证据窗口和检查事件顺序"],
        ["最终关系", "apple supported inside target；hand withdrawn", "末态图与释放后时间窗分别检查"],
        ["不变项", "两个碗不被移动；干扰物不参与", "独立追踪或定向复核；没看见不等于已证明没动"],
        ["来源", "真人窗口、sim episode、输入图、生成尝试的hash", "防串样、防重复计数、保留失败版本与数据划分"],
    ], [100, 330, CW - 430])
    r.p("任务规格是待满足的要求。任务审查VLM接收待观测关系与事件，结果仍需核验实际帧和时间引用。定向谓词会引导观察，不能把这种审查等同于不知道任务要求的盲评。", "body")
    r.p("仿真的控制阶段可帮助提取关键图，但“程序进入grasp阶段”本身不等于发生了有效接触；接触真值应由仿真状态另行核验。", "small")

    r.page("04  第一批实际结果：出片、可读和准入分开统计")
    r.table([
        ["候选", "实际看到的过程", "尚未解决", "完整gate"],
        ["黑碗左→右", "抓取、抬起、搬运、放下、撤手，角色方向正确", "未证明全片无移碗或倾倒", "needs_review"],
        ["白碗右→左", "修复后初态稳定，完成搬运并撤手", "禁止事件覆盖；末态支撑区间表达问题", "needs_review"],
        ["闭合书→开盒", "闭合书放入盒内；局部补帧后抓握可见", "未证明全片无开书／移盒", "needs_review"],
        ["只开右柜门", "正确开右门、释放、撤手", "额外手；柜内物体无法确认", "needs_review"],
    ], [100, 260, 295, CW - 655], size=10)
    r.p("四段均为24Hz、193帧、960×960，末帧PTS为8.0秒，媒体总时长约8.042秒。机器人仍是原生20Hz，二者没有逐帧动作对齐。")
    r.p("本批共16个付费阶段请求：3次编辑、3次视频提交、10次审核。轮询恢复没有再次提交视频；本报告不据此猜测账户账单金额。")
    r.p("3条任务过程可读候选 ≠ 3条完整自动验收通过。柜门的子动作正确，也不能覆盖其余对象和演员约束失败。")

    r.page("05  一条可读候选的完整链：苹果换碗", "真实人类任务 → 仿真执行A → 合成人类参考 → 独立机器人执行B")
    r.image_row([
        r.video_frame(real(black), 120, "A. 真人：将苹果从占用碗搬到空碗"),
        r.video_frame(donor(black), 151, "B. 仿真donor：黑碗，图像左→右"),
        r.video_frame(generated(black), 100, "C. 合成人类：保留黑碗任务角色"),
    ], height=188)
    r.image_row([
        r.video_frame(generated(black), 0, "生成初态：苹果仍在左侧源碗"),
        r.video_frame(generated(black), 192, "生成末态：右侧目标碗中，手已撤离"),
        r.video_frame(query(black), 163, "独立query：白碗另一执行；用其自身动作监督"),
    ], height=155)
    r.link_video("apple_black_generated.mp4", generated(black), "苹果黑碗：完整生成视频")

    r.page("06  一个真实修复例：早期苹果瞬移与角色混淆", "同一任务、历史失败版本与修复版本；不是随机多次试验的成功率估计")
    old_apple = ROOT / "pilot_20261020/seedance_recovery/candidate_generated_human.mp4"
    fixed_apple = ROOT / "repair_study_20261020_v1/role_grounded.mp4"
    r.image_row([
        r.video_frame(old_apple, 0, "旧视频：0秒，苹果位于右侧源碗"),
        r.video_frame(old_apple, 3, "旧视频：0.125秒，手接触前苹果跳到左侧"),
        r.video_frame(old_apple, -1, "旧视频末态：方向与原定任务角色不符"),
    ], height=160)
    r.image_row([
        r.video_frame(fixed_apple, 0, "修复后：角色从初图绑定，右源→左目标"),
        r.video_frame(fixed_apple, 72, "修复后：手抓持搬运，苹果没有先跳换碗"),
        r.video_frame(fixed_apple, 192, "修复后：左侧目标碗，释放并撤手"),
    ], height=160)
    r.p("本例生成侧修改的是固定对象角色和接触前状态的提示；审核侧增加了密集边界帧。一次修复成功只能证明存在可改善的实例，不能证明提示词已经解决隐藏状态和物理接触。", "small")
    r.link_video("apple_original_failure.mp4", old_apple, "失败版本")
    r.link_video("apple_role_grounded_repair.mp4", fixed_apple, "修复版本")

    r.page("07  接触过程：书本例里先补观测，再谈修视频")
    r.image_row([
        r.video_frame(generated(book), 68, "帧68 / 2.833秒：第一次审核区间起点"),
        r.video_frame(generated(book), 76, "帧76 / 3.167秒：补帧后建立抓握里程碑"),
        r.video_frame(generated(book), 79, "帧79 / 3.292秒：第一次审核区间终点"),
    ], height=196)
    r.table([
        ["环节", "实际处理", "得到什么"],
        ["初次审查", "24帧全片覆盖；抓握证据unknown", "定位到68–79帧区间"],
        ["预算选择", "保留原24帧，补入69–78共10帧", "总34帧，区间连续；没有重生成视频"],
        ["再次审查", "提交实际时间戳和连续帧", "抓握事件已建立；必要步骤全部匹配"],
        ["剩余限制", "禁止开书／移盒需要更大覆盖", "仍为needs_review，不伪造全片否定证据"],
    ], [100, 350, CW - 450], size=10)
    r.link_video("book_generated.mp4", generated(book), "书本入盒完整视频")

    r.page("08  柜门失败暴露了首图条件的根本不足")
    r.image_row([
        r.video_frame(donor(cab), 0, "仿真初态：柜门关闭，内部物体被遮挡"),
        r.video_frame(donor(cab), 498, "仿真末态：右门打开，可见内部彩色物体"),
    ], height=180)
    r.image_row([
        r.video_frame(generated(cab), 0, "旧合成初态：计划外多手，初图综合审查漏检"),
        r.video_frame(generated(cab), 192, "旧合成末态：门已开，但柜内物体身份不明确"),
    ], height=172)
    r.p("首图没有显示柜内物体，生成器不能从缺失的视觉信息恢复其身份。文字“不要移除物体”只表达约束，不能提供物体外观与位置。暗腔不能直接判为物体消失，也不能判为物体保留。", "small")
    r.link_video("cabinet_baseline.mp4", generated(cab), "柜门基线：首图＋文字")

    r.page("09  隐藏物体：加条件编译器和可见性记录",
           "本轮柜门用显式隐藏约束＋已有仿真末帧落地；通用遮挡推理和自动对象可见性账本尚未完成")
    r.table([
        ["模块", "输入与触发条件", "输出／处理"],
        ["状态条件编译", "任务要求保留某对象，但初帧看不见；或动作会暴露内部", "导出初态与显露后关键图；记录实体身份和可见区间"],
        ["能力适配", "生成接口明确支持相应首尾图或参考条件", "将关键图以实际支持的role传入；不把拼图当作时序控制"],
        ["关键图一致性", "同一柜体、内含物、相机、人物计划跨图", "先比对象数量、外观和位置；不一致则修关键图"],
        ["输出可见性检查", "生成中应出现打开后的柜内视野", "不可见→证据不足；可见但不符→内容失败"],
        ["纠错", "条件缺失／输出对象变化／帧间错误分别路由", "补条件、重新编辑或拒绝；每次保留父版本"],
    ], [120, 330, CW - 450], size=10.4)
    r.p("关键区别：审查模型知道“想要什么”还不够，视频生成模型也必须实际收到对应视觉条件。首尾图只限制两个端点，仍不能保证中途开门、接触与物体轨迹正确。")
    r.p("仿真资产的真实3D状态可以为仿真端提供可见性与对象身份依据；生成端的像素必须重新核验，不能借用仿真隐藏真值直接宣布正确。", "small")

    r.page("10  人物数量：生成前检查可见手数量")
    anchor = JOBS / cab / "human_initial.png"
    tool_dir = ROOT / "state_conditioned_20261020c/tools"
    has_local_hands = (tool_dir / "old_cabinet_hand_overlay.jpg").exists()
    r.image_row([
        {"path": str(tool_dir / "old_cabinet_hand_overlay.jpg") if has_local_hands else str(anchor),
         "label": "旧图：目视3手；MediaPipe容量设6仍只检出2手"},
        {"path": str(anchor), "crop": [650, 240, 1024, 560],
         "label": "原图局部放大：右侧两手交叠；检测器漏掉下面一手"},
    ], height=200)
    r.p("渲染计划先声明可见手数量预算。本例只开右柜门，修复方案为1只活动手；这不是管线固定单臂限制，双臂任务应按角色另设。VLM清点逐手位置，本轮另实装MediaPipe交叉检查，检测上限设6，避免截掉额外手。")
    r.table([
        ["输入图", "检测到的手／CPU耗时", "正确处理"],
        ["旧柜门初图", "2只 / 36.7毫秒，漏掉交叠手", "inconclusive；不能用≤2宣告通过"],
        ["修复后单手初图", "1只 / 25.7毫秒", "仍与VLM／人工证据组合，不单独放行"],
    ], [170, 295, CW - 465], size=9.8)
    r.p("专项VLM在旧图也只确认最少2只、精确数不确定。增加工具的价值是独立证据与暴露漏检；当前不能宣称已获得可靠的人体真值。耗时不含首次导入，模型初始化另约0.106秒。", "small")
    r.web_link("工具能力来源：Google MediaPipe Hand Landmarker 官方说明", "https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker")

    r.page("11  接触：图像证据与物理真值是两件事")
    r.table([
        ["检查层", "可借助工具", "可作出的结论"],
        ["事件定位", "VLM＋仿真阶段标签＋实际PTS", "找到应看哪里；阶段标签不是接触真值"],
        ["短区间连续观察", "OpenCV／FFmpeg取连续原帧；预算选择器", "接触前→抓握→共同运动→释放的视觉链是否可见"],
        ["对象与手的运动一致性", "后续可接分割／点追踪／光流；需置信度和遮挡状态", "检测异常跃迁或分离；仅2D，不自动变成3D接触"],
        ["仿真端物理", "对象状态、接触记录、关节与动作原生日志", "机器人仿真接触可验证；不直接认证生成的人类视频"],
        ["生成端不可判定", "保持unknown，另取角度／补条件／拒绝高要求用途", "不以VLM自报confidence代替真实标注"],
    ], [130, 330, CW - 460], size=10.2)
    r.p("对训练任务参考而言，至少要看出谁操作谁、往哪里、什么时候释放。若要把生成视频用于腕部3D或动力学监督，还需要另一套独立可验证的三维标注；本管线目前没有这些GT。")
    r.p("接触类错误需先区分：视频本来正确但未采到接触帧；接触被遮挡无法知道；视频真的发生悬浮或穿透。三者对应补帧、保留未知、内容修复，不能统一追加一句“物理正确”。", "small")
    r.web_link("可选分割工具（未部署）：SAM 2 官方仓库", "https://github.com/facebookresearch/sam2")
    r.web_link("可选点追踪工具（未部署）：CoTracker 官方仓库", "https://github.com/facebookresearch/co-tracker")

    diag_dir = ROOT / "state_conditioned_20261020c/tools/old_cabinet_contact"
    if (diag_dir / "diagnostic.json").exists():
        diagnostic = load(diag_dir / "diagnostic.json")
        r.page("本轮工具实测  时间展开补上静态检查的盲点", "固定相机柜门例；人工在首帧框定把手与静态背景，未使用虚构的生成端3D真值")
        r.image_row([
            {"path": str(diag_dir / "diagnostic_frame0000.jpg"),
             "label": "首帧：2个手候选；18个把手点＋36个背景点"},
            {"path": str(diag_dir / "diagnostic_frame0077.jpg"),
             "label": "后续帧：手分离后可检出3个手候选"},
            {"path": str(diag_dir / "diagnostic_frame0192.jpg"),
             "label": "末帧：把手点丢失，明确标unknown，不当作静止"},
        ], height=163)
        r.table([
            ["实际模块", "实际结果", "不能推出的结论"],
            ["MediaPipe逐帧6手容量", "193帧中169帧检出3手；首次0.583秒，最后7.583秒", "仍是候选检测，需要图像核验；不能自动推断人数"],
            ["KLT双向误差＋背景平移补偿", "手约1.58秒靠近；主要把手运动约3.33–5.83秒", "2D邻近和共运动不是3D接触或抓握真值"],
            ["有界追踪，不重新附着新点", "168帧有支持，最后25帧unknown；总CPU约9.69秒", "不能证明最后的放手／稳定，也不能保证遮挡后身份不漂移"],
        ], [200, 335, CW - 535], size=9.8)
        r.p("本例已经证明：只审核静态初图会漏掉重叠手，而逐帧时间展开能提供新的反例证据。ROI为人工首帧指定；要规模化，仍需把对象定位与置信度接入任务规格或分割模块。", "small")
        r.p("原始逐帧结果：tools/old_cabinet_contact/diagnostic.json、timeline.csv、point_evidence.json；阈值是本次诊断配置，尚未校准为数据验收标准。", "small")

    if new:
        add_new_case(r, new, args.case_manifest)

    r.page("12  四类故障怎样触发，而不是人工想起才处理",
           "路由规则已程序化；执行器只运行已实现分支，其余停止。真实执行记录见本轮改进页")
    r.table([
        ["四类问题与触发信号", "对应模块与路由动作", "继续执行的边界"],
        ["①接口状态／审查JSON格式：pending、未解决POST、schema形状不符",
         "provider状态适配；只读恢复task ID；保真object→array包装或停止",
         "不重复收费提交，不补真值／时间／证据，不把格式修复当内容改善"],
        ["②整体审核漏多手：超过本例手预算，或数量不明确",
         "actor专项VLM＋本地手候选；必要时局部和逐帧复核；修初图后重新审核",
         "通过前不提交视频；检测器漏检不能独立放行"],
        ["③稀疏帧错过接触：必要事件unknown，存在可定位短区间",
         "refinement预算选择＋连续原帧重审；KLT／手关键点辅助定位",
         "缺证据先补观察；明确瞬移／穿透是内容失败，不能靠补帧改成成功"],
        ["④首帧缺隐藏状态：对象需要保留但初帧不可见",
         "显露／终态条件准备＋payload实际role核验＋关键图对象检查",
         "能力不足或关键图不合格时停止，不能退回纯文字声称条件等价"],
    ], [250, 295, CW - 545], size=10)
    r.p("白碗的真实协议例：第一次回复缺少字段，保留拒绝；后续回复字段完整但把事件数组写成对象，仅8处做object→[object]包装，逐项核对内容hash，0次新API。未改语义，原失败记录仍保留。", "small")
    r.p("文件hash、窗口越界与来源不符属于所有阶段的基础校验；训练参考配对、缓存和动作时钟是独立的数据接口问题，见后续专页。")
    r.p("有界闭环需要明确最大图像编辑次数、视频提交次数、追加审核帧数与API阶段预算。调用前写入请求指纹，未解决的付费POST不自动重发；轮询可以沿原task ID恢复。", "small")

    r.page("13  框架与工具：每个组件承担什么工作")
    r.table([
        ["层", "目前使用", "用途与边界"],
        ["编排／持久化", "Python、JSON manifest、请求指纹、阶段产物目录", "可恢复的有界流水线；不依赖复杂agent框架才能实现"],
        ["仿真与来源", "既有human_to_robot_pipeline、原生arrays/receipt", "任务规划、机器人执行与本体动作留在原生端"],
        ["图像／视频模型", "Nano Banana编辑；Seedance生成；Gemini审查", "三者的输出均需验收，模型名称不等于能力证明"],
        ["媒体证据", "FFmpeg/ffprobe、OpenCV、PIL", "完整解码、实际PTS、精确帧窗、裁图与连续补帧"],
        ["确定性约束", "JSON Schema、task_evidence、显式事件匹配与偏序", "约束结构和证据引用；不从代码规则凭空生成视觉事实"],
        ["数据接口", "training_pairs隔离适配器、CPU实际读取和生产函数数值比对", "机器人观测与动作同源；合成参考不附虚构动作GT"],
        ["本地手实例", "MediaPipe＋local_hands.py，独立环境已实测", "提供检测下界；旧图漏检交叠手，不能单独通过"],
        ["后续可选", "SAM 2／CoTracker，以及仿真depth/mask导出", "尚未部署接入；未运行的工具不计入本轮成果"],
    ], [110, 335, CW - 445], size=9.8)
    r.p("repair_router.py只给出下一步计划，不调用模型或批准质量；本轮执行器才实际调度受支持分支。方法重点是任务规格决定所需条件、检查窗口与修复动作，程序负责来源、预算、版本和准入。", "small")

    r.page("14  数据接口：四个缺陷已在隔离适配器修复")
    r.table([
        ["旧路径风险", "新路径的具体约束", "实际验证"],
        ["按同goal扩大参考池，忽略指定边", "显式pair_id与允许query；不隐式替换参考", "4条跨执行边逐条读取"],
        ["合成来源枚举、准入与split追踪缺失", "synthetic_human、原始lineage、两层发布标志", "候选inspection可读，training全部拒绝"],
        ["缓存身份不含视频实际内容", "video hash＋精确窗口＋encoder设置参与key", "修改媒体身份应触发新缓存"],
        ["动作offset／fps混用", "robot原生20Hz、offset=0；human24Hz独立", "首／中／末12次读取，末端padding正确"],
    ], [235, 330, CW - 565], size=10.2)
    r.p("实际张量：参考32×3×224×224；query RGB为3×224×224；动作窗口16×20，其中14维有效；state为20维，其中18维有效。动作只从query B的原生数组和normalizer计算。")
    r.p("与未改动的生产媒体/EEF函数对比：RGB逐像素相同，归一化action最大绝对差1.14×10⁻⁷，state最大差4.33×10⁻⁷。这些是数值接口误差，不是生成的人手运动误差。")
    r.p("旧生产loader尚未整体切换。隔离读取成功不等于准入通过，更不等于已经有策略训练收益。", "small")

    r.page("15  真正用于模型训练时，数据怎样组成")
    r.table([
        ["网络／监督", "输入来自哪里", "为什么这样安排"],
        ["HyperNet参考", "合成人类视频A：任务正确且审核可追溯", "提供目标、对象关系、必要步骤；不要求逐帧等速"],
        ["PolicyNet观察", "独立机器人执行B的当前相机帧", "场景、布局、资产可以不同，避免复制生成donor"],
        ["机器人action", "同一执行B的原生动作；原始fps和offset", "保持图像—动作配对，不能拿A的动作监督B"],
        ["未来视觉辅助", "仅从已通过相应用途审核的视频构建前缀→未来", "伪影会成为错误监督；要先做对象和事件连续性检查"],
        ["人类腕部3D／接触力", "当前合成视频无可靠GT；原EgoDex数值本轮未核验使用", "不能把原真人poses、机器人EEF或仿真接触迁移成合成视频真值"],
    ], [150, 300, CW - 450], size=10.5)
    r.p("苹果：黑碗生成参考配白碗机器人query，反向亦然。书本：旧书参考配新资产机器人query，保留显式任务等价关系，不改写goal。柜门：参考配另一布局query。")
    r.p("去重必须按真人session、真实窗口、sim rollout、唯一初态和生成尝试分别计数。同一来源生成100个视频不会变成100段独立的人类示范。", "small")

    r.page("16  质量分级：让数据保留价值，同时不掩盖未知")
    r.table([
        ["级别", "至少需要的证据", "可以用于什么"],
        ["可解码候选", "媒体完整、时钟与来源一致", "检查与展示；不能因此进入训练"],
        ["任务参考候选", "角色、关键事件、结果可读；明确保留未知", "进一步人工／独立审核"],
        ["任务参考准入", "针对参考用途的必要约束完整通过，污染风险可接受", "任务条件训练；仍不宣称3D物理真值"],
        ["时序辅助准入", "接触前后与对象连续性达到预定覆盖要求", "未来视觉／时序关系监督"],
        ["运动／动力学真值", "独立可靠坐标、时序、对应与接触测量", "需要额外标注和验证；当前生成片段没有"],
    ], [150, 350, CW - 500], size=10.5)
    r.p("“没看到错误”与“有足够证据验证正确”应分开。完整禁止事件验收可能要求全片覆盖；任务参考准入可以制定专门规范，但必须事先定义、独立评估，不能为这四条样例临时降低标准。")
    r.p("用于任务参考学习，并不要求每段视频都拥有真实3D或接触力标签；应验证相应用途需要的可见任务语义与时序质量。本轮尚未做独立质量发布，也没有启动训练。", "small")
    r.p("稳定末态也要区分单帧支持关系与连续一段时间的稳定支持。旧白碗出现的是证据区间表达与末态窗口不匹配；不能删掉早期观测来制造通过。", "small")

    r.page("17  继续扩展时：最需要做的受控实验")
    r.table([
        ["对照", "控制变量", "优先指标"],
        ["首图文字 vs 关键状态条件", "同任务、同源sim、同生成模型、同总调用预算", "对象保存、关键事件、有效任务参考产率"],
        ["综合审核 vs 专项＋有界补帧", "同一候选集、独立人工判定、同审核预算", "漏检率、误拒率、unknown率、每例审核成本"],
        ["随机重试 vs 错误类型路由", "同样失败池、同样最大生成次数", "修复率、成本、产生新错误的比例"],
        ["同一初态不同目标", "布局与物体相同，任务目标／顺序不同", "生成后目标差异是否仍存在"],
        ["独立布局和资产", "真实任务逻辑保持、视觉因素变化", "跨场景对应与下游参考依赖性"],
    ], [215, 330, CW - 545], size=10.2)
    r.p("应先积累分任务、分失败类型的小批量盲评，再判断哪些模块值得扩大。当前三类任务、单次修复不能估计总体提升，也不能比较生成模型的可靠性排名。")
    r.p("下游训练实验放在质量稳定之后：同预算原数据、增加合格配对、增加未筛选配对三组，区分有效任务信息与纯数据量。当前没有启动这些训练。", "small")

    r.page("18  论文价值应落在可复验的机制与证据")
    r.bullets([
        "任务关键状态由任务规格和仿真可观测状态共同决定，不按等间隔随意挑图；隐藏内容需要被真正传给生成器。",
        "审核围绕事件而非只评价整片好不好；遇到缺证据时精确补帧，遇到内容错误时才重新生成。",
        "修复动作由失败类型和剩余预算决定；保留失败版本、输入、调用、证据与成本，使“有效配对产率”可以复现。",
        "跨执行配对和同初态不同目标防止只看场景猜任务；后续才验证HyperNet是否确实依赖参考。",
    ])
    r.p("这些组成一个有研究价值的方向，但不能仅凭组合工具宣称方法新颖或优于既有工作。需要与首帧编辑、关键帧条件、视频编辑和自动数据筛选等既有方法做同预算对照。")
    r.p("本报告已经展示的证据主要是具体故障、接口可读性和局部修复案例。端到端训练提升、真实机器人迁移、通用物理正确性尚未建立。", "small")

    r.page("19  复现与文件索引", "PDF中的视频入口是本地附件链接；没有把PDF静态页伪装成可播放视频")
    r.p("报告目录", "sub")
    r.p(str(Path(args.output).parent), "small")
    r.p("videos/：完整视频普通文件，可点击相对链接。VIDEO_INDEX.md：原文件绝对路径与SHA256。assets/：真实解码帧。report_provenance.json：逐页、逐帧来源与报告hash。", "body")
    r.table([
        ["证据", "文件位置"],
        ["基础批次输入", str(BATCH / "input_manifest.json")],
        ["基础批次统计", str(BATCH / "BATCH_REPORT.json")],
        ["完整四角色视频", str(BATCH / "review_bundle")],
        ["CPU读取快照", str(BATCH / "training_pair_inspection_snapshots/ac81e404ed146b5638ca")],
        ["新增修复结果", args.case_manifest or "本次导出仅含已完成基础批次；没有加入新增结果"],
        ["架构与结果原文", "data_generation_methods/PIPELINE_ARCHITECTURE_20261020B.md\nPIPELINE_BATCH_RESULTS_20261020B.md\nSTATE_CONDITIONED_RESULTS_20261020C.md"],
    ], [140, CW - 140], size=9.5)
    verification_file = Path(args.output).parent / "evidence/software_verification.json"
    if verification_file.exists():
        verification = load(verification_file)
        r.p(f"软件验证：{verification['passed']}项测试通过，用时{verification['elapsed_seconds']}秒。测试未调用远端模型，验证接口与规则，不代表相同数量的视频成功或物理验证。", "small")
    r.p("拷贝给他人时保留整个report目录与videos/相对位置即可。PDF不包含API凭据、请求认证头或中转站密钥。", "small")
    provenance = {
        "batch_manifest": {"path": str(BATCH / "input_manifest.json"), "sha256": sha(BATCH / "input_manifest.json")},
        "batch_report": {"path": str(BATCH / "BATCH_REPORT.json"), "sha256": sha(BATCH / "BATCH_REPORT.json")},
    }
    if args.case_manifest:
        provenance["new_case_manifest"] = {"path": args.case_manifest, "sha256": sha(args.case_manifest)}
        evidence_dir = Path(args.output).parent / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        existing_evidence_index = (evidence_dir / "INDEX.json").exists()
        if not existing_evidence_index:
            shutil.copy2(args.case_manifest, evidence_dir / "CASE_REPORT.json")
        case_dir = Path(new["new_video"]).parent
        for source, target in [
            (case_dir / "video_review_v1/interpretation.json", "video_review_interpretation.json"),
            (case_dir / "video_review_v1/visible_evidence_summary.json", "video_visible_evidence_summary.json"),
            (case_dir / "condition_review_v1/decision.json", "condition_review_decision.json"),
            (case_dir / "generated_media.json", "generated_media.json"),
            (case_dir / "source_manifest_erratum.json", "source_manifest_erratum.json"),
            (case_dir / "dispatcher_v1/STATUS.json", "dispatcher_STATUS.json"),
            (case_dir / "dispatcher_v1/execution_journal.jsonl", "dispatcher_execution_journal.jsonl"),
        ]:
            if source.exists():
                if not existing_evidence_index:
                    shutil.copy2(source, evidence_dir / target)
                provenance[target] = {"path": str(source), "sha256": sha(source)}
        for name in ("old_cabinet_contact", "new_cabinet_contact_v1"):
            source = ROOT / "state_conditioned_20261020c/tools" / name / "diagnostic.json"
            if source.exists():
                if not existing_evidence_index:
                    shutil.copy2(source, evidence_dir / f"{name}_diagnostic.json")
                provenance[name] = {"path": str(source), "sha256": sha(source)}
        keyframes_dir = Path(args.output).parent / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        keyframe_records = []
        for index, item in enumerate(new.get("inputs", []) + new.get("edited_keyframes", [])):
            source = Path(item["path"])
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            target = keyframes_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            keyframe_records.append({"label": item["label"], "source": str(source), "file": str(target.relative_to(Path(args.output).parent)), "sha256": sha(source)})
        (keyframes_dir / "manifest.json").write_text(json.dumps(keyframe_records, ensure_ascii=False, indent=2))
    r.save(provenance)
    print(json.dumps({"pdf": str(r.output), "pages": r.page_no, "frames": len(r.frame_records), "videos": len(r.video_records)}, ensure_ascii=False))


def add_new_case(r, new, manifest_path):
    """The root writes only observed new-case facts to this external manifest."""
    r.page("本轮改进 A  实际给了生成器哪些新条件", new.get("title", "柜门关键状态条件实验"))
    inputs = new.get("inputs", [])
    images = []
    for item in inputs:
        if "frame_index" in item:
            images.append(r.video_frame(item["path"], item["frame_index"], item["label"]))
        else:
            images.append(item)
    if images:
        r.image_row(images[:3], height=210)
    r.p(new.get("input_explanation", "下图是实际使用的输入材料；具体角色与阶段以请求产物为准。"))
    if new.get("condition_rows"):
        r.table([["条件", "实际使用方式", "能力边界"]] + new["condition_rows"], [160, 330, CW - 490], size=10)
    if new.get("input_notes"):
        for text in new["input_notes"]:
            r.p(text, "small")

    r.page("本轮改进 B  中间关键图与生成前检查")
    edited = list(new.get("edited_keyframes", []))
    if len(edited) == 2 and Path(new.get("new_video", "")).parent.name == "cabinet_case":
        edited.append({
            "path": edited[1]["path"], "crop": [490, 340, 635, 495],
            "label": "同一终图局部放大：上层彩色块；仅裁剪显示，未重画",
        })
    if edited:
        r.image_row(edited[:3], height=215)
    r.p(new.get("anchor_explanation", "中间关键图先检查对象身份、场景、人物数量和任务状态，再决定是否调用视频生成。"))
    rows = new.get("anchor_outcomes", [])
    if rows:
        r.table([["检查项", "实际结果", "证据／限制"]] + rows, [150, 300, CW - 450], size=10)

    r.page("本轮改进 C  新视频的实际过程")
    video = new.get("new_video")
    if video and Path(video).exists():
        indices = new.get("video_frame_indices", [0, 38, 76, 115, 153, -1])
        items = [r.video_frame(video, idx) for idx in indices]
        r.image_row(items[:3], height=164)
        if len(items) > 3:
            r.image_row(items[3:6], height=164)
        r.p(new.get("video_observation", "图示来自实际输出视频，是否满足任务由后页的逐项证据决定。"), "small")
        r.link_video("cabinet_state_conditioned.mp4", video, "本轮状态条件生成结果")
    else:
        r.p(new.get("video_observation", "本轮没有取得可供展示的新视频；流程停在已记录的失败阶段。"))
        for x in new.get("failure_details", []):
            r.p(x)

    r.page("本轮改进 D  变化与仍然缺少的证据")
    outcomes = new.get("outcomes", [])
    if outcomes:
        r.table([["问题", "原版本", "本轮结果", "证据／局限"]] +
                [[x["item"], x["before"], x["after"], x.get("evidence", "")] for x in outcomes],
                [105, 180, 230, CW - 515], size=9.5)
    interpretation_file = Path(new.get("new_video", "")).parent / "video_review_v1/interpretation.json"
    if interpretation_file.exists():
        interpretation = load(interpretation_file)
        if interpretation.get("all_ten_visible_queries_returned_yes"):
            r.p("输出审核：Gemini查看24帧，10项可见问题均答yes。该结果只支持所查帧中的任务过程；自由文本中的“确认物理交互”等越界断言不被采纳，原回复和解释记录分别保留。", "small")
    if new.get("counts"):
        if isinstance(new["counts"], list):
            r.table([["本轮统计", "实际值"]] + new["counts"], [250, CW - 250], size=9.5)
        else:
            r.p("本轮统计：" + json.dumps(new["counts"], ensure_ascii=False), "small")
    for text in new.get("limitations", [])[:4]:
        r.p(text, "small")
    r.p("“独立VLM复核”仅指另一次请求查看实际输出帧，并非已证明误差独立。图像编辑与审核都来自Gemini系列，可能存在相关偏差，需要额外人工或异构工具验证。", "small")
    r.p("原始记录：" + str(manifest_path), "small")
    if new.get("modules"):
        r.page("本轮改进 E  自动分支实际触发记录", "只列真实执行状态；可调用接口与无人值守自动执行不是一回事")
        r.table([["模块", "触发条件", "本轮实际状态"]] +
                [[x["name"], x["trigger"], x.get("status", "") + "；" + x.get("result", "")] for x in new["modules"]],
                [170, 300, CW - 470], size=10)
        r.p(new.get("dispatcher_explanation", "新初图重编辑由人工明确发起；其后终态条件、条件核验和视频提交按dispatcher的条件分支执行。每步保留输入事实、计划、执行前后产物及停止原因；仅有计划不算已执行。"), "body")
        if new.get("remaining_automation"):
            r.p("自动化边界：" + new["remaining_automation"], "small")
        status_file = Path(new.get("new_video", "")).parent / "dispatcher_v1/STATUS.json"
        if new.get("dispatcher_status_explanation"):
            r.p(new["dispatcher_status_explanation"], "small")
        elif status_file.exists():
            state = load(status_file)
            r.p("主dispatcher记录：" + str(state.get("router_state"))
                + "；下一动作：" + str(state.get("next_action", {}).get("kind"))
                + "；结果：" + str(state.get("outcome", {}).get("reason", state.get("outcome", {}).get("kind")))
                + "。执行范围应结合上表真实handler记录判断，不能把未接通的计划算作完成。", "small")
        r.p("本轮改变了初图、可见手计划和终态条件。只有一个实例，不能将观察到的改善单独归因于首尾帧，也不能据此估计整体成功率提升。", "small")
    old_diag = ROOT / "state_conditioned_20261020c/tools/old_cabinet_contact"
    new_diag = ROOT / "state_conditioned_20261020c/tools/new_cabinet_contact_v1"
    if (old_diag / "timeline.png").exists() and (new_diag / "timeline.png").exists():
        r.page("本轮改进 F  新旧局部交互时间线", "横轴均为各自视频实际时间；不是两段视频逐帧动作对齐")
        r.image_row([
            {"path": str(old_diag / "timeline.png"), "label": "旧视频：末段缺点→保留unknown；时间展开发现额外手"},
            {"path": str(new_diag / "timeline.png"), "label": "新视频：先靠近，后门动，末段手分离而把手趋稳"},
        ], height=262)
        r.table([
            ["工具指标", "旧视频", "新视频"],
            ["有足够把手跟踪点的帧", "168 / 193；末25帧unknown", "193 / 193"],
            ["手候选数量", "169帧检出3只", "193帧均检出1只"],
            ["本地CPU主循环", "9.69秒", "6.84秒"],
        ], [250, 260, CW - 510], size=9.7)
        r.p("上图从上至下分别为：背景补偿后把手像素位移、最近手关键点距离、存活跟踪点／手候选数。灰虚线只是本次近邻诊断阈值。CPU耗时不含整条API生成流程。", "small")
        r.p("新旧图像和人工ROI不同，跟踪点充足不等于感知性能提升；KLT通过也可能跟到遮挡物。所有曲线只作为2D可见交互证据，不作为3D接触、施力或抓握真值。", "small")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(OUT / "Real2Sim2Human_pipeline_20261020.pdf"))
    p.add_argument("--font", default=str(OUT / "assets/NotoSerifSC-Regular.ttf"))
    p.add_argument("--case-manifest")
    build(p.parse_args())
