"""Build the revised mechanism-focused presentation from audited local evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from kai_deck import Deck, ROOT, INK, MUTED, TEAL, BLUE, RED, GOLD, LINE, WHITE, sha

W=Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi')
OLD=Path('/mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021')
CONTENT=json.loads((OLD/'presentation/content_final.json').read_text())
C=CONTENT['cases']
MV=Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview')
ASSET=Path('/mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_assets_v3')

def notes(*paths,extra=''):
    rows=[]
    for p in paths:
        p=Path(p)
        if not p.is_absolute():p=ROOT/p
        assert p.exists(),p
        rows.append({'path':str(p),'sha256':sha(p)})
    return json.dumps({'evidence':rows,'interpretation':extra},ensure_ascii=False,indent=2)

def footer(d,text,color=TEAL,y=6.54):
    d.text(.74,y,11.86,.48,text,18,color,True)

def mini_contract(d,x,y,w,h):
    d.rect(x,y,w,h,'F1F5F1',LINE,True)
    if w<2:
        for i,s in enumerate(['roles','order','goal']):
            d.rect(x+.1,y+.13+i*.25,.06,.13,TEAL,TEAL)
            d.text(x+.23,y+.08+i*.25,w-.31,.22,s,11,BLUE,True)
        return
    d.text(x+.12,y+.09,w-.24,.3,'TASK CONTRACT',13,TEAL,True)
    for i,(a,b) in enumerate([('roles','apple / source / target'),('order','grasp → lift → release'),('goal','inside target + stable')]):
        yy=y+.49+i*.34
        d.text(x+.12,yy,.67,.26,a,11,BLUE,True)
        d.text(x+.82,yy,w-.9,.28,b,10,INK)

def mini_guard(d,x,y,w,h):
    d.rect(x,y,w,h,'F1F5F1',LINE,True)
    if w<2:
        for i,(s,c) in enumerate([('Task',TEAL),('Object',RED),('Actor',GOLD)]):
            d.rect(x+.1,y+.12+i*.25,.08,.13,c,c)
            d.text(x+.24,y+.08+i*.25,w-.3,.22,s,11,c,True)
        return
    for i,(label,state,col) in enumerate([('任务目标','支持',TEAL),('对象身份','失败',RED),('人物归属','待确认',GOLD)]):
        yy=y+.17+i*.4
        d.text(x+.12,yy,w-1.0,.3,label,13,INK)
        d.text(x+w-.85,yy,.75,.3,state,12,col,True)

def cover(d):
    d.page('Real → Sim → Human：机制、现状与简化方案',
           '把任务语义保留下来；相机、人物站位和机器人实现可以合理改变。',
           notes(OLD/'delivery.json',extra='New explanatory names map existing responsibilities; no new production modules claimed.'),'OVERVIEW')
    imgs=[(C[0]['human_input'],'真实任务：EgoDex 窗口'),(C[0]['sim_frame'],'仿真：对象角色与物理执行'),(C[1]['output_after'],'合成人类：局部任务已可读')]
    for i,(p,caption) in enumerate(imgs):
        x=.75+i*4.18;d.rect(x,1.88,3.8,2.9,WHITE,LINE,True);d.picture(p,x+.15,2.02,3.5,2.25,caption)
        if i<2:d.label_arrow(x+3.86,3.1,x+4.08,3.1)
    d.card(.75,5.12,3.8,1.11,'数据可以扩','已有大规模本地索引；已审任务仍少。',TEAL,17)
    d.card(4.93,5.12,3.8,1.11,'默认更轻','清晰桌面优先固定对面机位。',TEAL,17)
    d.card(9.11,5.12,3.48,1.11,'检查围绕失败','人物、隐藏物体、关键事件分开。',TEAL,17)
    footer(d,'本轮重新审计已有输入与结果；没有新收费生成，也没有把建议写成已验证效果。',MUTED,6.6)

def pipeline(d):
    d.page('六个环节，把实际输入、模型与输出连起来',
           '六个英文名是演示简称；模型名按实际 API 别名标注，不代表已核验底层权重。',
           notes(W/'data_generation_methods/r2h_pipeline/human_task_draft.py',W/'data_generation_methods/r2h_pipeline/providers.py',W/'data_generation_methods/r2h_pipeline/repair_router.py','quality_repair/inventory.json',extra='Actual relay IDs: gemini-3.1-pro-preview; gemini-nano-banana-2.1; doubao-seedance-2-5-260628. Backend weights independently unverified. Gemini blind extraction was exercised once; existing reviewed task specs largely came from assistant review and explicit code templates. Labels are explanatory, not new implementations.'),'PIPELINE')
    cards=[
        (.72,1.86,'CLIP','选完整任务窗口','索引召回＋agent 确认\n角色与动作证据','Gemini 3.1 Pro：单例草案支线',C[0]['human_input']),
        (4.93,1.86,'PACT','冻结任务合同','初态／顺序／终态／不变量','Python 模板＋schema 校验',None),
        (9.14,1.86,'CAST','调用已适配技能','角色需求 → 资产、布局与任务技能','本地 catalog＋任务 adapter',C[0]['sim_frame']),
        (9.14,4.13,'SYNC','执行并留下证据','动作／状态／相机共享时间轴','ManiSkill / SAPIEN / MPlib',MV/'development/visual_evidence/close_opposite_0078.png'),
        (4.93,4.13,'HUM','条件图与人类视频','首／终条件图 → 视频\n单人新图待验证','Nano Banana 2.1＋Seedance 2.5',C[1]['output_after']),
        (.72,4.13,'SENTRY','分项检查、定位修复','工具初筛 → 局部审核 → 配对','OpenCV / FFmpeg＋Gemini＋agent',None),
    ]
    for x,y,name,title,body,model,img in cards:
        d.rect(x,y,3.48,2.02,WHITE,LINE,True)
        d.text(x+.15,y+.13,1.45,.39,name,23,TEAL,True)
        d.text(x+1.54,y+.13,1.8,.51,title,14,INK,True)
        if img:d.picture(img,x+.15,y+.65,1.16,.81)
        elif name=='PACT':mini_contract(d,x+.15,y+.59,1.2,.94)
        else:mini_guard(d,x+.15,y+.59,1.2,.94)
        d.text(x+1.45,y+.68,1.86,.8,body,14)
        d.text(x+.15,y+1.65,3.2,.28,model,11,MUTED)
    for x in [4.28,8.49]:d.label_arrow(x,2.86,x+.55,2.86)
    d.label_arrow(10.88,3.94,10.88,4.06)
    for x in [9.03,4.82]:d.label_arrow(x,5.15,x-.55,5.15)
    footer(d,'通过 → 保存来源与配对；失败 → 回到对应条件或技能。视角不要求与真人视频一致。',TEAL,6.6)

def sources(d):
    d.page('数据源：并不缺视频，已确认的完整任务窗口还很少',
           '当前实际使用 EgoDex；索引不等于已解码，HOT3D 另有 1,028 个归档路径缺失。',
           notes('source_contract/source_contract_evidence.json'),'CLIP')
    d.table(.75,1.88,[3.18,2.36,6.29],[
        ['本地统一索引','用途行数','这些数意味着什么'],
        ['EgoDex','141,391','141,359 个原始 clip 身份；不是已审任务数'],
        ['HOT3D','3,832','有索引；未进入当前三任务合成人类试验'],
        ['ImItator','810','episode 记录对应 53 个媒体身份'],
        ['合计','146,033','含用途重复；145,244 个媒体身份'],
    ],row_h=.67,size=16)
    d.card(.75,5.48,5.67,.83,'实际采用','冻结 11 个任务窗口；旧 pilot 使用其中 8 个 train 窗口。',TEAL,16)
    d.card(6.64,5.48,5.94,.83,'视频实验范围','苹果、书本入箱、柜门：3 个来源窗口、3 类任务。',TEAL,16)
    footer(d,'可以扩量：先同机制换来源／对象，再新增接触机制。Zero-WAM 的 2,499 配对是公开清单。',MUTED,6.58)

def clip(d):
    d.page('CLIP：角色看状态关系，窗口覆盖完整的原子任务',
           '现在以索引召回＋agent 看图确认起步；VLM 产出草案，代码检查结构，不能自动盖章。',
           notes('source_contract/source_contract_evidence.json',extra='Apple and book reviews were Codex visual review, not user manual annotations. One actual Gemini cabinet draft: 24 sampled frames from an externally chosen 72-frame clip, about30.128s.'),'CLIP')
    for i,(p,title,body) in enumerate([
        (C[0]['human_input'],'① 对象角色','苹果＝被搬物；初始盛苹果的碗＝源；另一个空碗＝目标。'),
        (C[2]['human_input'],'② 窗口筛选','从可辨初态，覆盖抓取、搬运、释放，到结果稳定；缺一步就换窗口。'),
        (C[3]['human_input'],'③ 语义确认','检查“开门”还是“开门后取物”；不能从长任务随意截一个好看的前缀。')]):
        x=.75+i*4.04;d.rect(x,1.86,3.74,4.42,WHITE,LINE,True);d.picture(p,x+.18,2.03,3.38,1.66)
        d.text(x+.18,3.98,3.38,.45,title,20,TEAL,True);d.text(x+.18,4.61,3.38,1.23,body,18)
    footer(d,'已做的 VLM 盲提取仅 1 个柜门窗口：Gemini 3.1 Pro 看 24 帧，约 30 秒，输出仍是 draft。',MUTED,6.59)

def pact(d):
    d.page('PACT：合同写“必须发生什么”，不锁死人类相机与动作',
           '以下以苹果换碗为例：真实来源绑定＋对象角色＋事件和状态约束。',
           notes('source_contract/source_contract_evidence.json',W/'data_generation_methods/human_to_robot_pipeline/planning_contract.py',extra='Visibility VLM label below is a proposed extension, not present in the frozen production contract.'),'PACT')
    rows=[
        ('来源与窗口','视频哈希、帧区间、fps、来源组和数据划分'),
        ('对象角色','被搬苹果、源碗、目标碗；非目标物保持不动'),
        ('初态与终态','苹果在源碗 → 同一苹果在目标碗内稳定'),
        ('必要顺序','抓持 → 抬离碗沿 → 搬运 → 释放 → 撤手'),
        ('禁止与允许','禁止换物／瞬移／漏步骤；允许视角和人手入场位置变化'),
        ('执行与证据','技能前后条件、失败码、状态／接触／来源绑定'),
    ]
    for i,(a,b) in enumerate(rows):
        x=.75+(i%2)*6.08;y=1.88+(i//2)*1.27
        d.card(x,y,5.73,1.1,a,b,TEAL,17)
    d.rect(.75,5.92,11.82,.64,'F5EFE3','E4D4B4',True)
    d.text(.95,6.07,11.43,.34,'建议补充：operator_count=1；visibility_vlm_check=0/1；触发原因与待查事件。',16,GOLD,True)
    footer(d,'目前是已审 spec → Python 编译合同。统一自动提取、资产能力与检查策略尚未完全接通。',MUTED,6.66)

def assets(d):
    d.page('CAST：本地资产已是万级，短板是“可用于哪种交互”',
           '同一物体的文件存在、能渲染、能当障碍物、能稳定操作，是不同层级。',
           notes('sim_assets/inventory_snapshot.json','sim_assets/household_tier_recheck.json','sim_assets/qualified_task_asset_scope.json',extra='Historical accepted robot recordings use their own mechanism-specific checks, not all qualified under latest strict apple planner. Corpus was not re-executed this round.'),'CAST')
    d.table(.75,1.88,[3.08,2.05,2.15],[
        ['模型来源','索引模型','文件存在'],
        ['Objaverse','46,305','46,305'],
        ['ABO','7,953','7,953'],
        ['GSO','1,033','1,033'],
        ['YCB','96','95'],
        ['Poly Haven','149','149'],
    ],row_h=.58,size=16)
    gallery=[('soda_can','易拉罐'),('tissue_box','纸巾盒'),('water_bottle','水瓶'),('soap_dispenser','皂液瓶'),('milk_carton','奶盒'),('spatula','锅铲'),('remote_control','遥控器'),('closed_jam_jar','果酱罐')]
    for i,(name,label) in enumerate(gallery):
        x=8.38+(i%2)*2.08;y=1.9+(i//2)*.87
        p=ROOT/'sim_assets'/f'actual_{name}_crop.png';d.picture(p,x,y,1.72,.55)
        d.text(x,y+.59,1.72,.25,label,12,MUTED,False,'center')
    d.text(.82,5.63,11.7,.4,'55,535 条模型路径现存，跨来源未去重；另有 2,537 个 RoboCasa MJCF 定义。',17,INK)
    d.text(.82,6.13,11.7,.3,'历史机器人库另有 6 个任务家族、3,337 条接受记录；不等于全部已通过最新 strict 规划资格。',14,MUTED)
    footer(d,'本轮家居适配：12 类预检、8 类静态实跑；这批扩增的新主动操作资格为 0。',TEAL,6.66)

def cast(d):
    d.page('CAST：先按能力选资产，再调用技能；不能只按名称找模型',
           '现有 catalog 与加载器可复用；建议增加“任务角色 → 交互能力”的检索层。',
           notes('sim_assets/inventory_snapshot.json',W/'data_generation_methods/human_to_robot_pipeline/household_assets.py',extra='Capability-aware retrieval is proposed; no generalized automatic task-to-assets loader is claimed.'),'CAST')
    stages=[('角色需求','杯子／碗／柜门\n需要执行什么交互'),('能力过滤','可抓部位、容腔\n铰链、尺寸、碰撞'),('资产登记','本地优先；缺失再取\n哈希、许可、尺度'),('预检与接入','几何／质量／关节\nloader → 技能适配')]
    for i,(a,b) in enumerate(stages):
        x=.75+i*3.05;d.card(x,1.98,2.68,1.69,a,b,TEAL,18)
        if i<3:d.label_arrow(x+2.76,2.8,x+2.97,2.8)
    d.table(.75,4.05,[2.15,4.57,5.1],[
        ['任务','机制必须具备的适配','只有漂亮网格为什么不够'],
        ['苹果入碗','有效容腔、碗沿净空、抓取与落点','整物凸包可能把碗口封死'],
        ['开柜门','被动铰链、门轴、把手接触与跟随','静态柜子没有真正可动的门'],
        ['灵巧手抓取','惯性、拇指对置、接触反馈','关节能动，不等于能抓稳'],
    ],row_h=.61,size=16)
    footer(d,'扩资产数量容易；可复用的能力标签、技能前后条件和资格缓存，才是扩任务的关键。',TEAL,6.65)

def sync(d):
    d.page('SYNC：不是额外生成一遍，而是给同次执行留下可复查记录',
           '动作＝发了什么，状态＝发生什么，相机＝能看见什么；多视角是可选记录方式。',
           notes(W/'data_generation_methods/human_to_robot_pipeline/multiview_recording.py','sim_assets/camera_evidence_summary.json'),'SYNC')
    labels=[('动作 a_t','关节／末端／夹爪命令'),('状态 s_t','对象位姿、抓持、接触、关节'),('相机 I_t','原视角＋对面视角及内外参')]
    for i,(a,b) in enumerate(labels):
        y=2.01+i*1.08;d.text(.85,y,2.2,.43,a,21,TEAL,True);d.text(.85,y+.47,3.7,.36,b,15)
        d.line(5.0,y+.4,12.0,y+.4,LINE,2)
        for j,tag in enumerate(['初态','抓持','搬运','释放','终态']):
            x=5.0+j*1.75;d.rect(x-.055,y+.32,.11,.16,TEAL,TEAL)
            if i==0:d.text(x-.36,y-.15,.8,.31,tag,14,MUTED,False,'center')
    d.card(.75,5.48,5.76,.85,'条件提取','从仿真状态取初态、关键显露态和终态。',TEAL,16)
    d.card(6.76,5.48,5.81,.85,'独立复查','另一检查器读取原始状态，核验碰撞、抓持、稳定。',TEAL,16)
    footer(d,'物理检查主要是几何与数值程序，不靠 AI 投票；双相机是同一条轨迹，不计两条新动作。',MUTED,6.6)

def camera(d):
    d.page('可见性按需：简单桌面优先对面机位',
           '建议合同字段 visibility_vlm_check=0/1；检查任务证据，不要求复制真人机位。',
           notes('sim_assets/camera_evidence_summary.json',extra='Risk label is a proposed opt-in policy. Existing acceptance standards are unchanged. Object pixel visibility is not proof of contact visibility.'),'SYNC')
    d.picture(MV/'development/visual_evidence/close_native_0078.png',.75,1.92,3.05,2.08,'原机位：闭爪关键区域被遮挡')
    d.picture(MV/'development/visual_evidence/close_opposite_0078.png',4.0,1.92,3.05,2.08,'对面机位：该布局目标更清楚')
    d.card(7.48,1.92,5.08,2.55,'已观察到的范围','两个同步苹果布局，对面机位闭爪阶段均为 16/16 帧可见。\n目标像素可见，不等于接触细节已被证明。',TEAL,18)
    d.card(.75,4.83,5.73,1.31,'默认 0\n免额外 VLM','已适配、清晰桌面、简单搬运；保留廉价画幅／目标像素检查。',TEAL,17)
    d.card(6.77,4.83,5.79,1.31,'触发 1\n查指定事件','拥挤遮挡、大物体密集、柜内显露、细小接触，或数值初筛异常。',GOLD,17)
    footer(d,'建议尚未接生产：0 只关闭额外可见性 VLM；物理、人物、对象身份检查照常。',MUTED,6.56)

def anchors(d):
    d.page('HUM：图像外观已经可用，人物结构和条件状态仍有硬伤',
           'Nano Banana 图像接口＋Seedance 视频接口；当前是关键帧条件生成，不是逐帧转写机器人动作。',
           notes('quality_repair/anchors_inventory.json','quality_repair/quality_matrix.json'),'HUM')
    anchor=json.loads((ROOT/'quality_repair/anchors_inventory.json').read_text());idx={x['id']:x['path'] for x in anchor}
    visuals=[(idx['batch_apple_initial'],'苹果首图：桌面、材质和物体清楚'),(idx['batch_book_initial'],'书本首图：画面自然，但两名操作者'),(idx['state_cabinet_initial'],'柜门首图：关闭状态无法给出内部物体')]
    for i,(p,cap) in enumerate(visuals):d.picture(p,.75+i*4.06,1.9,3.72,2.62,cap)
    d.card(.75,5.22,5.73,1.1,'视频实际输入','任务文本＋首图；B 组另给终图。整条动作／深度未输入。',TEAL,17)
    d.card(6.77,5.22,5.79,1.1,'锚图看图结果','5/5 初态可读，4/5 终态正确；人物 6 张错、4 张未知。',GOLD,17)
    footer(d,'应先确定单人站位、双手分工与必要状态图；新单人首图尚未生成验证。',MUTED,6.6)

def apple(d):
    d.page('例 1–2：错误苹果终图与视频末帧回跳一致',
           '同一来源：初次 B 与后验修复；状态差分来自 agent 看图确认，不是用户逐例标注。',
           notes(OLD/'evidence/case_media_manifest.json',extra='Repair modifies terminal image and resamples video simultaneously; does not isolate causal contribution. Both videos still violate single-operator requirement.'),'HUM · CASES')
    for row,c in enumerate(C[:2]):
        y=1.92+row*2.1;d.text(.75,y,2.35,.45,'初次：回跳' if row==0 else '修后：回跳消失',19,RED if row==0 else TEAL,True)
        d.picture(c['condition'],3.0,y,2.2,1.45,'终态条件')
        d.picture(c['output_before'],5.62,y,2.2,1.45,'第 191 帧')
        d.movie(c['output_video'],c['output_after'],8.26,y,2.2,1.45,'第 192 帧 · 播放')
        d.text(10.85,y+.23,1.65,1.16,'误回左碗' if row==0 else '留在右碗',19,RED if row==0 else TEAL,True)
    footer(d,'修复分支：核对源状态 → 修终图 → 重生成 → 同规则重审。两名操作者的问题仍未修复。',MUTED,6.54)

def book(d):
    d.page('例 3：书本入箱与撤手可读，人物检查必须独立列项',
           '窗口是“闭合书入已打开箱子”，不把它扩写成开箱、清空或整理整张桌子的任务。',
           notes(OLD/'evidence/case_media_manifest.json'),'HUM · CASES')
    c=C[2]
    for i,(p,cap) in enumerate([(c['human_input'],'真人窗口'),(c['sim_frame'],'仿真终态'),(c['condition'],'共同首图：两名操作者')]):d.picture(p,.75+i*3.03,1.94,2.67,2.62,cap)
    d.movie(c['output_video'],c['output_after'],9.85,1.94,2.67,2.62,'实际视频 · 播放')
    d.card(.75,5.23,5.73,1.08,'任务层面','入箱与撤手有局部可见支持。',TEAL,18)
    d.card(6.77,5.23,5.79,1.08,'人物层面','旧规则只数手，未判断是否属于同一人。',RED,18)
    footer(d,'所以可以说“任务过程表现不错”，但不能把它算作满足单人要求的有效配对。',MUTED,6.6)

def cabinet(d):
    d.page('例 4–5：柜门动作相似，隐藏物体的结果完全不同',
           '首图看不到的物体不能靠模型猜；审核需要对照源显露图与相对尺寸。',
           notes(OLD/'evidence/case_media_manifest.json','quality_repair/quality_matrix.json'),'HUM · CASES')
    d.picture(d.crop(C[4]['sim_frame'],[160,90,355,305]),.75,2.02,2.5,2.54,'仿真显露：柜内小方块')
    d.movie(C[3]['output_video'],C[3]['output_before'],3.67,2.02,2.5,2.54,'A：小方块变成大格板')
    d.picture(C[4]['condition'],6.62,2.02,2.5,2.54,'B：提供显露后的终图')
    d.movie(C[4]['output_video'],C[4]['output_before'],9.55,2.02,2.5,2.54,'B：小方块有局部保留')
    d.card(.75,5.2,5.73,1.17,'A 的反例','VLM 七项全 yes 仍漏检；同请求重抽也没修好。',RED,18)
    d.card(6.77,5.2,5.79,1.17,'B 的边界','开门、撤手和小方块局部可读；人物归属仍 unknown。',TEAL,18)
    footer(d,'重点查：物体身份、相对柜格比例、显露位置。不是“同颜色”就算保持了同一物体。',MUTED,6.62)

def quality(d):
    d.page('暂不计人数：多数已测短任务可读，尚无泛化结论',
           '新增清点去重：14 条真实生成视频，来自 3 类任务、3 个来源窗口；不是 14 个独立任务。',
           notes('quality_repair/quality_matrix.json','quality_repair/inventory.json',extra='Retrospective assistant visual review of 16 selected frames per video plus referenced prior dense checks. Not blind, not whole-video contact or continuity certification, not a newly promoted acceptance rate.'),'HUM · EVIDENCE')
    for i,(num,title,body,color) in enumerate([
        ('12 / 14','终态目标有支持','只问最后是否搬到／入箱／开门。',TEAL),
        ('9 / 14','局部任务综合有支持','兼看对象和已检查的连续性；暂不计人数。',TEAL),
        ('4 ＋ 1','明确反例 ＋ 未知','4 条方向／回跳／对象错误；1 条隐藏物体未知。',RED)]):
        x=.75+i*4.05;d.rect(x,1.98,3.74,2.46,WHITE,LINE,True);d.text(x+.2,2.17,3.34,.7,num,36,color,True,'center');d.text(x+.2,3.03,3.34,.4,title,20,color,True,'center');d.text(x+.23,3.6,3.28,.65,body,16,MUTED)
    d.table(.75,4.83,[3.12,3.18,5.52],[
        ['分组','局部任务支持','解释'],
        ['受控初轮 / 后验','4/6 ；1/2','不同条件与修复，分别保留分母'],
        ['更早探索','4/6','另 1 条失败、1 条未知；不能混作方法增益'],
    ],row_h=.54,size=16)
    footer(d,'本轮看图复核，不是全帧或盲测通过率。下一步必须换来源、换机制做保留验证。',MUTED,6.65)

def sentry(d):
    d.page('SENTRY：工具做可计算的检查，模型只回答有证据的问题',
           '验收不是一个总分，也不是 agent 看完说“不错”；每层输出 pass / fail / unknown。',
           notes(W/'data_generation_methods/r2h_pipeline/quality_cascade.py',W/'data_generation_methods/r2h_pipeline/contact_diagnostics.py',W/'data_generation_methods/r2h_pipeline/boundary_discontinuity.py',W/'data_generation_methods/r2h_pipeline/single_actor_acceptance.py'),'SENTRY')
    d.table(.75,1.92,[2.12,3.53,4.28,1.89],[
        ['层','当前工具／模型','具体检查','触发'],
        ['数据完整性','FFprobe / Python','解码、时间戳、帧数、来源哈希与划分','全部'],
        ['仿真物理','独立几何／状态验收器','碰撞、真实抓持、释放支撑与稳定','已适配执行'],
        ['条件图','Gemini/agent 观察＋规则门','人数归属、对象角色、初／终态','生成视频前'],
        ['视频事件','OpenCV / MediaPipe＋Gemini','抓／放／显露、边界跳变、对象比例','关键事件'],
        ['争议处理','密帧／局部裁剪＋agent','模型矛盾、未知、明确失败的修复分支','异常升级'],
    ],row_h=.70,size=15)
    footer(d,'工具与规则门已能运行；MediaPipe 只给可见手下界，通用语义检测和完整修复仍有缺口。',MUTED,6.49)

def detail(d):
    d.page('重要部位单独检查：先定位，再加密，不整条视频反复泛评',
           '模拟阶段知道事件时刻；合成视频先定位对应事件，不能硬套仿真帧号。',
           notes(W/'data_generation_methods/r2h_pipeline/refinement.py',W/'data_generation_methods/r2h_pipeline/hidden_identity_extent.py','quality_repair/quality_matrix.json'),'SENTRY')
    rows=[('人物结构','首图＋必要全景','数身体与手臂归属；两只手不等于一个人'),('隐藏对象','源显露图＋柜格局部','身份、形状、相对比例；同色不能替代身份'),('抓取／释放','定位事件前后密帧','接近→接触→共同移动→分离，缺证据保留未知'),('连续性','首尾密帧＋疑似跳变区','全局帧差初筛，再用背景配准／ROI 复核')]
    for i,(a,b,c) in enumerate(rows):
        y=1.94+i*1.07;d.rect(.75,y,11.82,.91,WHITE,LINE,True);d.text(.95,y+.18,2.07,.4,a,20,TEAL,True);d.text(3.18,y+.17,3.02,.48,b,18,BLUE);d.text(6.42,y+.16,5.87,.59,c,17)
    footer(d,'局部跟踪、尺寸与帧差是诊断信号；人类 RGB 没有真实 3D 接触力，不能由工具分数冒充。',MUTED,6.56)

def fix(d):
    d.page('修复要对应错误源；当前路由器会出计划，尚非通用自动修复器',
           '柜门专用 handler 已接；通用 router 仅出计划，未实现分支会停止，失败状态不等于自动重生成。',
           notes(W/'data_generation_methods/r2h_pipeline/repair_router.py',W/'data_generation_methods/r2h_pipeline/posthoc_controlled_repair.py',extra='Router emits plans, actions_executed=false. Historical apple repair was orchestrated with explicit visual state delta, not autonomous generalized repair. Segment generation is future work.'),'SENTRY · FIX')
    d.table(.75,1.92,[2.72,4.31,4.79],[
        ['失败类型','下一步做什么','实现与验证边界'],
        ['提交状态未知','查原 task ID；禁止盲目再提交','已有恢复／预算逻辑'],
        ['人数／身体错误','改首图与身体方案，重新审首图','规则已升级；新单人图尚未生成验证'],
        ['终态错／隐藏物体错','从源状态补显露或终图，再重生成','苹果局部有效；柜门仍有对象漏检'],
        ['接触证据不足','定位事件，加密帧和局部裁剪','工具可运行；仍需有约束的视觉判断'],
        ['规划／抓持失败','改资产或控制技能，不改视频提示词','双臂／灵巧手失败已留档'],
        ['长任务漏步骤','先确认时长能力，再考虑分段接续','后续方案，未完成实际通用验证'],
    ],row_h=.61,size=15)
    footer(d,'重抽只用于有理由的随机失败；“错误条件没改”或“同请求已重复失败”时不继续赌 seed。',MUTED,6.61)

def roadmap(d):
    d.page('下一步：把主路径做轻，把扩量集中到可复用机制',
           '先复用现有数据与工具，不再给每条简单样本叠加相同的泛化审核。',
           notes('source_contract/source_contract_evidence.json','sim_assets/camera_evidence_summary.json',extra='These are proposed priorities and interfaces, not newly qualified default implementations.'),'DESIGN')
    d.card(.75,1.97,3.74,3.95,'01  同机制先扩来源','苹果／书本先换真人窗口、对象实例与布局。\n\n按源视频／执行分组划分，保留未参与调试的验证集。',TEAL,20)
    d.card(4.8,1.97,3.74,3.95,'02  资产能力与技能对接','为资产登记：可抓／有容腔／有铰链／可倒液等能力和已验证用途。\n\n匹配不到技能就明确未支持。',TEAL,20)
    d.card(8.85,1.97,3.74,3.95,'03  检查按风险升级','简单场景固定对面机位；额外可见性 VLM 默认不调用。\n\n人物、对象、关键事件失败各走不同分支。',TEAL,20)
    footer(d,'优先指标：每次尝试的有效配对、每条有效数据的时间／实扣成本；不是模块数量或 VLM 分数。',MUTED,6.53)

def costs(d):
    d.page('时间与费用：保留原实测，不把简化设计写成已经省了多少',
           '下面是既有受控批次的口径；本轮仅做本地审计与 PPT，没有新增生成调用。',
           notes(OLD/'evidence/timing.json',OLD/'evidence/timing_cpu_supplement.json',OLD/'evidence/cost_evidence.json'),'APPENDIX')
    d.table(.75,1.92,[3.44,2.45,5.93],[
        ['阶段','实测时间','边界'],
        ['已适配苹果仿真','30–34 秒','两次后来的同步多相机执行'],
        ['首图／终图','25–34 / 36–46 秒','首图为 HTTP；终图为完整操作'],
        ['单条视频','212–275 秒','提交至下载，含排队／轮询'],
        ['VLM／本地工具','20–54 / 1–10 秒','本地工具分通用质量和边界检查'],
        ['六条、三组并行','10 分 53 秒','不含旧仿真、共同首图和人工筛选'],
    ],row_h=.6,size=16)
    d.text(.85,5.82,11.52,.7,'已查服务商报价；历史逐条视频的人民币实扣仍未知。审核条件估算约 ¥0.18–1.33/次，不能当视频价格。',18,BLUE,True)
    d.text(.85,6.65,11.5,.29,'报价依据：api.echoflowai.cc/pricing；原始价格字段与计费假设见备注及上版证据。',12,MUTED,url='https://api.echoflowai.cc/pricing')

def controls(d):
    d.page('控制补充：双臂同步与灵巧手已实跑，稳定抓持仍需改进',
           '这些是已保存的上一轮控制结果，本轮未新增 GPU 实验。',
           notes(OLD/'dual_arm/ppt_results.json',OLD/'dexterous/evidence_summary/qualification_status.json',OLD/'control_independent/imitator_backend_audit.json'),'APPENDIX')
    dc=CONTENT['controls'][0];hc=CONTENT['controls'][2]
    dv=next(x for x in dc['visuals'] if x.get('video'));hv=next(x for x in hc['visuals'] if x.get('video'))
    d.movie(dv['video'],dv['image'],.85,1.95,5.3,2.8,'双 planner → 同步 20 Hz → 两果同抬')
    d.movie(hv['video'],hv['image'],6.88,1.95,5.3,2.8,'Franka＋Allegro：Imitator Game 实际执行')
    d.text(.85,5.33,5.3,.92,'双臂同时运动 4.45 秒；四次小样完整验收 0/4。串行对照也失败，不能报提速比。',18)
    d.text(6.88,5.33,5.3,.92,'23 维控制、15 秒轨迹；实际推块后空抬。源模型 13 处惯性违例，先修资产再谈抓稳。',18)
    footer(d,'旧生产默认和验收阈值未修改；接口接通、局部有效、稳定操作资格分开汇报。',MUTED,6.6)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    d=Deck(a.out)
    for f in [cover,pipeline,sources,clip,pact,assets,cast,sync,camera,anchors,apple,book,cabinet,quality,sentry,detail,fix,roadmap,costs,controls]:f(d)
    print(d.save())

if __name__=='__main__':main()
