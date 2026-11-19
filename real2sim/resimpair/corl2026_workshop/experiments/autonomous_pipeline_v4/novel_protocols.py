"""Pre-execution mechanism contracts evaluated by the unchanged interpreter."""
import math
from . import planning_protocols as p
from .event_protocol import validate


def protocol(scene,task):
    selected=set(task["object_ids"]);ids=[o["id"] for o in scene["objects"]]
    joints=scene["joints"];all_ids=ids+[j["id"] for j in joints]
    entities=[{"id":o["id"],"description":o["semantic_description"],"kind":o["shape"],
        "role":"target" if o["id"] in selected else "unchanged","geometry":o["geometry"],"pose_channel":o["id"]+".pose"} for o in scene["objects"]]
    entities += [{"id":j["id"],"description":j["semantic_description"],"kind":j["joint_type"],
        "role":"target" if j["id"] in selected else "unchanged",
        "geometry":{k:j[k] for k in ["pivot_world","axis_world","limits","parts","spring_N_or_Nm"]},
        "pose_channel":j["id"]+".pose"} for j in joints]
    entities.append({"id":"table","description":"tabletop","kind":"plane","role":"support","geometry":{}})
    plan=p.base(task["id"],task["family"],entities,task["required_skills"],12)
    poses,grasps=p.rigid_channels(plan,ids)
    contacts={}
    for arm in range(2):
        for i,oid in enumerate(all_ids):
            name=f"arm{arm}.{oid}.contact"
            contacts[arm,oid]=p.channel(plan,name,"robot_contact_impulses",[arm,i],"contact_impulse",[oid])
            plan["effectors"][f"arm{arm}"]["contact_channels"][oid]=name
    q={};v={}
    for i,j in enumerate(joints):
        oid=j["id"]
        q[oid]=p.channel(plan,oid+".q","cabinet_qpos",[i],"joint_position",[oid])
        v[oid]=p.channel(plan,oid+".dq","cabinet_qvel",[i],"joint_velocity",[oid])
        p.channel(plan,oid+".pose","door_world",[i],"transform",[oid])
        plan["initial_conditions"].append(p.condition(oid+".initial","joint_range",[oid],p.expr("lt",p.expr("abs",q[oid]),p.const(.01)),range=[-.01,.01]))
        if oid not in selected:
            p.invariant(plan,oid+".untouched","joint_range",[oid],p.expr("lt",p.expr("abs",q[oid]),p.const(.01)),range=[-.01,.01])
    for oid in ids:
        if oid not in selected:p.unchanged(plan,oid,poses[oid])
    cross=p.channel(plan,"robots.contact","cross_robot_force_N",[],"contact_force",[])
    p.invariant(plan,"robot_collision","no_contact",[],p.expr("lt",cross,p.const(.5)),visible=False)
    kind=task["scene_kind"];param=task["planning_parameters"]
    if joints:
        previous=None
        chosen=param.get("selected",[0])
        for index in chosen:
            oid=joints[index]["id"];arm=index if len(joints)>1 else 0
            start=oid+".touch"
            p.event(plan,start,"trigger",[oid],p.expr("gt",contacts[arm,oid],p.const(1e-6)),f"arm{arm}",after=[previous] if previous else [])
            if kind=="spring_console":
                pressed=oid+".pressed"
                p.event(plan,pressed,"articulate",[oid],p.expr("gt",q[oid],p.const(.012)),f"arm{arm}",after=[start])
                goal=p.expr("lt",p.expr("abs",q[oid]),p.const(.004))
                previous=oid+".rebound"
                p.event(plan,previous,"release",[oid],goal,f"arm{arm}",after=[pressed],strict=[pressed])
                parameters={"rest_range_m":[0,.004],"required_press_travel_m":.012}
            else:
                direction=param.get("direction",1 if index==0 else -1)
                target=.075 if kind=="linear_faders" else direction*(.85 if kind=="rotary_indexer" else .30 if kind=="balance_beam" else .45)
                tolerance=.022 if kind=="linear_faders" else .16 if kind=="rotary_indexer" else .12
                arrival=p.expr("lt",p.expr("abs",p.expr("sub",q[oid],p.const(target))),p.const(tolerance))
                previous=oid+".actuated"
                p.event(plan,previous,"articulate",[oid],arrival,f"arm{arm}",after=[start])
                if param.get("return_to_origin"):
                    goal=p.expr("lt",p.expr("abs",q[oid]),p.const(.08))
                    back=oid+".return"
                    p.event(plan,back,"articulate",[oid],goal,f"arm{arm}",after=[previous],strict=[previous]);previous=back
                    target=0.
                else:goal=arrival
                parameters={"target_joint_position":target,"tolerance":.08 if param.get("return_to_origin") else tolerance}
            plan["goals"] += [p.condition(oid+".goal","joint_range",[oid],goal,**parameters),
                              p.condition(oid+".settled","stable",[oid],p.expr("lt",p.expr("abs",v[oid]),p.const(.06)))]
    elif kind=="rolling_lanes":
        previous=None
        for index in param["selected"]:
            oid="ball_"+str(index);arm=index;position=p.expr("position",poses[oid])
            touch=oid+".push"
            p.event(plan,touch,"trigger",[oid],p.expr("gt",contacts[arm,oid],p.const(1e-6)),f"arm{arm}",after=[previous] if previous else [])
            angular=p.channel(plan,oid+".spin","object_states",[ids.index(oid),{"slice":[10,13]}],"angular_velocity",[oid])
            roll=oid+".rolling"
            p.event(plan,roll,"arrive",[oid],p.expr("gt",p.expr("norm",angular),p.const(.3)),f"arm{arm}",after=[touch])
            y=next(o["position"][1] for o in scene["objects"] if o["id"]==oid)
            inside=p.expr("all",p.expr("gt",p.component(position,0),p.const(.14)),p.expr("lt",p.component(position,0),p.const(.27)),
                          p.expr("lt",p.expr("abs",p.expr("sub",p.component(position,1),p.const(y))),p.const(.045)),
                          p.expr("lt",p.component(position,2),p.const(.075)))
            previous=oid+".captured";p.event(plan,previous,"arrive",[oid],inside,after=[roll])
            plan["goals"].append(p.condition(oid+".goal","inside",[oid],inside,region_x_m=[.14,.27],lane_center_y_m=y))
            p.invariant(plan,oid+".never_grasped","no_grasp",[oid],p.expr("not",p.expr("any",grasps[0,oid],grasps[1,oid])))
    else:
        tool={"hook_retrieval":"hook","granule_sweeping":"paddle","contact_drafting":"stylus","bead_stirring":"stirrer"}[kind]
        held=grasps[0,tool]
        p.event(plan,"tool.grasp","grasp",[tool],held,"arm0")
        p.invariant(plan,"tool.held","continuous_grasp",[tool],held,start="tool.grasp",effector="arm0")
        plan["terminal_hands"]["arm0"]={"state":"holding","entity":tool}
        plan["goals"].append(p.condition("tool.held","held",[tool],held))
        previous="tool.grasp"
        if kind in {"hook_retrieval","granule_sweeping"}:
            cargo_ids=(["cargo_"+str(k) for k in param["selected"]] if kind=="hook_retrieval" else
                       [f"grain_{k}_{i}" for k in param["selected"] for i in range(3)])
            for oid in cargo_ids:
                collision=p.channel(plan,tool+"."+oid+".hit","object_contact_impulses",[ids.index(tool),ids.index(oid)],"contact_impulse",[tool,oid])
                event_id=oid+".contact"
                p.event(plan,event_id,"passive_contact",[tool,oid],p.expr("gt",collision,p.const(1e-6)),after=["tool.grasp"])
                position=p.expr("position",poses[oid]);initial=next(o["position"] for o in scene["objects"] if o["id"]==oid)
                lo,hi=(-.20,-.065) if kind=="hook_retrieval" else (.10,.29)
                inside=p.expr("all",p.expr("gt",p.component(position,0),p.const(lo)),p.expr("lt",p.component(position,0),p.const(hi)),
                              p.expr("lt",p.expr("abs",p.expr("sub",p.component(position,1),p.const(initial[1]))),p.const(.09)),
                              p.expr("lt",p.component(position,2),p.const(.08)))
                p.event(plan,oid+".goal","arrive",[oid],inside,after=[event_id])
                plan["goals"].append(p.condition(oid+".goal","inside",[oid],inside,region_x_m=[lo,hi]))
                for arm in range(2):p.invariant(plan,f"arm{arm}.{oid}.no_direct_touch","no_contact",[oid],p.expr("le",contacts[arm,oid],p.const(1e-5)),effector=f"arm{arm}")
        else:
            support="drawing_board" if kind=="contact_drafting" else "mixing_bowl"
            tip=p.channel(plan,tool+".tip","tool_tips_world",[ids.index(tool)],"position",[tool])
            hit=p.channel(plan,tool+".surface_contact","object_contact_impulses",[ids.index(tool),ids.index(support)],"contact_impulse",[tool,support])
            points=([x+[param["height"]] for x in param["points"]] if kind=="contact_drafting" else
                    [[-.10+.052*math.cos(a),-.18+.052*math.sin(a),.026] for a in
                     [param["direction"]*2*math.pi*k/12 for k in range(12*param["turns"]+1)]])
            for k,point in enumerate(points):
                near=p.expr("lt",p.expr("norm",p.expr("sub",tip,p.const(point))),p.const(.018 if kind=="bead_stirring" else .012))
                if kind=="contact_drafting":near=p.expr("all",near,p.expr("gt",hit,p.const(1e-7)))
                eid="trace."+str(k)
                p.event(plan,eid,"arrive",[tool,support],near,"arm0",after=[previous],strict=[previous]);previous=eid
            plan["goals"].append(p.condition(support+".preserved","unchanged_pose",[support],
                p.expr("lt",p.expr("norm",p.expr("sub",p.expr("position",poses[support]),p.expr("position",p.expr("initial",poses[support])))),p.const(.002))))
            if kind=="bead_stirring":
                for i in range(6):
                    oid="bead_"+str(i);position=p.expr("position",poses[oid])
                    inside=p.expr("all",p.expr("lt",p.expr("norm",p.expr("sub",p.expr("xy",position),p.const([-.10,-.18]))),p.const(.115)),
                                  p.expr("lt",p.component(position,2),p.const(.10)))
                    plan["goals"].append(p.condition(oid+".contained","inside",[oid],inside))
                    if i<2:
                        c=p.channel(plan,oid+".stir_contact","object_contact_impulses",[ids.index(tool),ids.index(oid)],"contact_impulse",[tool,oid])
                        p.event(plan,oid+".agitated","passive_contact",[tool,oid],p.expr("gt",c,p.const(1e-6)),after=["tool.grasp"])
    from .hand_correspondence import declare_from_scene
    declare_from_scene(plan,scene)
    validate(plan)
    return plan
