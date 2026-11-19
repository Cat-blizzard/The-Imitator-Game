"""Build the concise evidence presentation from reviewed local content JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from slide_builder import Deck, BLUE, RED, GREEN, GRAY, INK, LIGHT


def intro(d):
    d.page('Real → Sim → 合成人类参考：当前做到哪一步',
           '五次真实运行 · 分阶段耗时与费用 · 双臂与灵巧手实测',
           '范围：五例来自三个历史开发任务。所有视觉结果保留原始失败与后验修复。'
           '最新同步相机、严格规划及候选人体模型不能追溯赋予旧视频新的物理资格。'
           '本次成本查询不触发付费生成。新控制实验另行记录。')
    d.text(.65,1.68,12,1.0,'物理轨迹可以批量生成；\n合成人类参考仍需分项验收和针对性修复。',25,bold=True)
    d.table(.65,3.05,[2.65,4.5,4.85],[
        ['环节','已有实际证据','当前边界'],
        ['严格苹果仿真','15 次执行 / 6 组动态轨迹','不代表其他接触任务通过'],
        ['合成人类视频','3 任务 / 8 条历史输出','双人、错误终态、隐藏对象失真'],
        ['本次展示','5 次运行 + 新控制小样','不将局部成功计为全链合格'],
    ],row_h=.73,size=15)
    d.text(.7,6.35,11.85,.5,'优先解决：人物与对象条件正确、关键交互可见、错误能被独立检出。',18,BLUE,bold=True)


def pipeline(d):
    d.page('当前 pipeline：哪些已自动化，哪些仍需要工程与审查',
           '人物几何与隐藏状态编译仍为候选；各阶段资格分别判断，训练准入保持关闭。',
           '完整实现目录见 START_HERE.md / CURRENT_STATE.json。任务适配和资产/控制机制仍需人工工程输入。'
           '同步多相机已在两苹果布局实跑；人物几何与隐藏内容条件编译为 opt-in 候选，未证明合成人类质量改善。')
    boxes=[
        (.65,1.75,'真实人类任务窗口','完整动作、对象角色\n窗口筛选 / 语义确认'),
        (3.84,1.75,'结构化任务合同','目标、必要顺序、不变量\nVLM 提议 + 来源核对'),
        (7.03,1.75,'机器人仿真任务','资产、布局、控制与规划\n任务机制仍需适配'),
        (10.22,1.75,'动作 / 状态 / 多视角','同一次执行共享时间轴\n独立物理与可见性检查'),
        (10.22,4.15,'合成人类条件图','单人站位 / 左右手\n初态 / 显露 / 终态'),
        (7.03,4.15,'合成人类视频','首图 + 文本 / 可选终图\n8 秒视频逐次 API 生成'),
        (3.84,4.15,'分项验收与修复','人物 / 对象 / 接触 / 连续性\n失败定位，unknown 保留'),
        (.65,4.15,'可追溯数据配对','来源、执行、相机、划分\n合格后才准入'),
    ]
    bw=2.47
    for x,y,title,body in boxes:
        d.rect(x,y,bw,1.6)
        d.text(x+.12,y+.17,bw-.24,.63,title,16,BLUE,bold=True)
        d.text(x+.12,y+.86,bw-.24,.62,body,11.5)
    for x in [.65,3.84,7.03]:d.line(x+bw+.08,2.55,x+3.19-.08,2.55,GRAY,1.2,True)
    d.line(11.45,3.45,11.45,4.05,GRAY,1.2,True)
    for x in [10.22,7.03,3.84]:d.line(x-.08,4.95,x-3.19+bw+.08,4.95,GRAY,1.2,True)
    d.text(.7,6.26,11.9,.65,'循环：哪里首次引入错误 → 修对应条件或控制 → 同规则重审；不靠重抽或提高 VLM 评分代替验证。',17,BLUE)


def case_page(d,c,i):
    d.page(f'例 {i}  {c["title"]}',c['subtitle'],c['notes'])
    # Input and simulator are deliberately separate from generated conditions.
    for x,w,title in [(.6,3.7,'输入 → 仿真'),(4.55,2.3,'中间条件'),(7.15,5.55,'实际输出')]:
        d.text(x,1.58,w,.38,title,18,BLUE,bold=True)
    d.picture(c['human_input'],.6,2.05,1.75,2.5,'真实任务窗口')
    sim=d.crop(c['sim_frame'],c['sim_crop']) if c.get('sim_crop') else c['sim_frame']
    d.picture(sim,2.55,2.05,1.75,2.5,c.get('sim_caption','语义对应仿真'))
    d.picture(c['condition'],4.55,2.05,2.3,2.5,c['condition_caption'])
    before=d.crop(c['output_before'],c['before_crop']) if c.get('before_crop') else c['output_before']
    d.picture(before,7.15,2.05,2.6,2.5,c['before_caption'])
    d.movie(c['output_video'],c['output_after'],10.05,2.05,2.5,2.5,c['after_caption']+' · 点击播放')
    d.line(.63,5.15,12.65,5.15,LIGHT)
    d.text(.65,5.43,1.05,.35,'结果质量',16,c.get('quality_color',RED),True)
    d.text(1.85,5.41,10.7,.7,c['quality'],16)
    d.text(.65,6.22,1.05,.35,'外部输入',16,BLUE,True)
    d.text(1.85,6.20,10.7,.7,c['intervention'],15)


def interventions(d):
    d.page('为什么“给一个 VLM 提示词”还不能完成这条管线',
           '区分开发者工程输入、助手看图复核与用户逐例标注；历史人工总工时尚未记录。')
    d.table(.65,1.75,[2.35,4.7,4.95],[
        ['需要什么','当前真实做法','单次提示不足的原因'],
        ['任务与资产适配','选完整窗口；确认角色；补控制技能','模型文字不保证资产可交互、路径可达'],
        ['身体与手的安排','一个身体、左右手分工；首图先检查','两只手可能来自两个人'],
        ['状态修复','苹果案例经看图确认左源右目标，再改终图','错误条件会传到视频；不能只催模型重做'],
        ['对象身份与接触','对照原物体、柜格比例、显露与接触帧','VLM 全 yes 仍可漏掉小方块变大面板'],
        ['独立资格检查','动作、物理、人物、对象、连续性分开','脚本成功 / 视频下载完成 ≠ 有效配对'],
    ],row_h=.78,size=14)
    d.text(.7,6.65,11.9,.32,'已新增自动条件编译与来源绑定；生成后的视觉正确性仍需实际验证。',14,BLUE)


def timing(d,t):
    d.page('运行时间：已适配苹果仿真约半分钟，视频数分钟',
           t['subtitle'],t['notes'])
    d.table(.65,1.7,[3.0,2.25,6.75],t['rows'],row_h=.56,size=13)
    d.text(.7,6.3,11.9,.64,t['conclusion'],17,BLUE,bold=True)


def cost(d,c):
    d.page('费用：已查到服务商配置，实际扣费与报价分开',
           c['subtitle'],c['notes'])
    d.table(.65,1.73,[2.5,4.3,5.2],c['rows'],row_h=.8,size=14)
    d.text(.7,6.15,11.9,.78,c['conclusion'],16,BLUE,bold=True)
    d.text(.7,6.94,11.9,.18,'当前报价来源：api.echoflowai.cc/pricing  |  估算假设、原始计时与逐项证据见备注',8.5,GRAY,
           url='https://api.echoflowai.cc/pricing')


def bottlenecks(d):
    d.page('当前最影响 pipeline 的问题',
           '按实际失败与覆盖缺口排序；没有证据表明时间或费用是唯一主要瓶颈。')
    items=[
        ('1  有效产率：人物与对象条件容易失真',
         '双人首图会传给全部派生视频；隐藏物体会换形、变大；错误终图会把视频拉回错误状态。',RED),
        ('2  审核可信度：看起来对、VLM 说对，都不够',
         '要分别核查人数归属、物体身份与比例、关键接触、连续性；证据不足时保留 unknown。',RED),
        ('3  覆盖与工程适配：新接触机制仍需要技能',
         '苹果规划通过不代表柜门、杯盖或灵巧手通过；多视角和外观变体不能算作新动作。',BLUE),
        ('4  延迟：每条视频约数分钟，失败修复会重复等待',
         '先审首图 / 终图再发视频；批量仿真能摊薄初始化，但不能消除生成和审查时间。',BLUE),
    ]
    for j,(title,body,col) in enumerate(items):
        y=1.7+j*1.25
        d.text(.7,y,11.85,.4,title,19,col,True)
        d.text(.72,y+.49,11.75,.58,body,16)


def controls(d,c):
    for item in c:
        d.page(item['title'],item['subtitle'],item['notes'])
        if item.get('kind')=='inventory':
            d.table(.65,1.72,[3.05,2.2,6.75],item['rows'],row_h=.75,size=14)
            d.text(.7,6.45,11.9,.5,item['limits'],16,BLUE,bold=True)
            continue
        for j,visual in enumerate(item['visuals']):
            x=.7+j*4.17
            if visual.get('video'):
                d.movie(visual['video'],visual['image'],x,1.8,3.8,2.65,visual['caption'])
            else:
                img=d.crop(visual['image'],visual['crop']) if visual.get('crop') else visual['image']
                d.picture(img,x,1.8,3.8,2.65,visual['caption'])
        d.text(.7,5.08,11.9,.74,item['result'],18,item.get('result_color',BLUE),True)
        d.text(.7,5.99,11.9,.89,item['limits'],16)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--content',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();content=json.loads(Path(a.content).read_text())
    d=Deck(a.out);intro(d);pipeline(d)
    assert len(content['cases'])==5
    for i,c in enumerate(content['cases'],1):case_page(d,c,i)
    interventions(d);timing(d,content['timing']);cost(d,content['cost']);bottlenecks(d)
    controls(d,content['controls'])
    print(d.save())


if __name__=='__main__':main()
