"""Ten functional scene recipes, each with three task branches.

Procedural geometry IS the native collision geometry. No fluid, dust removal,
electrical operation, or real household-asset fidelity is claimed. These are
new mechanism probes, not re-labelled translations of the previous blocks.
"""
from copy import deepcopy
import math

from .core import require
from .library import CAMERAS

COLORS = {"red": [.78,.15,.12,1.], "blue": [.12,.35,.78,1.], "yellow": [.92,.66,.12,1.],
          "gray": [.34,.39,.43,1.], "wood": [.55,.36,.18,1.], "white": [.85,.87,.85,1.],
          "green": [.18,.57,.37,1.], "dark": [.13,.16,.19,1.]}
KINDS = ["spring_console", "rocker_console", "linear_faders", "rotary_indexer", "balance_beam",
         "rolling_lanes", "hook_retrieval", "granule_sweeping", "contact_drafting", "bead_stirring"]
NOVELTY = {
 "spring_console": ("elastic_press", "press a spring-return plunger and observe displacement plus rebound"),
 "rocker_console": ("nonprehensile_rocker", "press opposite sides of a pivoting rocker to change its angular state"),
 "linear_faders": ("constrained_slide", "move controls along physical rails; no lifting or free-space relocation"),
 "rotary_indexer": ("tangential_rotation", "apply tangential contact to a vertical-axis indexed dial"),
 "balance_beam": ("lever_equilibrium", "tilt and return a mass-balanced beam about its support"),
 "rolling_lanes": ("rolling_capture", "roll selected spheres along bounded lanes into terminal pockets"),
 "hook_retrieval": ("tool_mediated_pull", "grasp an L hook, engage the far face of a cargo, then pull without direct hand-cargo contact"),
 "granule_sweeping": ("tool_mediated_sweep", "push selected loose rigid granules with a wide paddle into a collection region"),
 "contact_drafting": ("contact_path_trace", "trace different connected contours with a physically grasped stylus contacting a board"),
 "bead_stirring": ("confined_tool_agitation", "move a held paddle around a bowl while making measured contact with loose beads"),
}


def box(oid, position, half, color="gray", *, movable=False, mass=.04, parts=None, description=None):
    return {"id":oid,"position":list(position),"half_size":list(half),"yaw":0.,
            "shape":"compound_boxes" if parts else "box","kind":"movable" if movable else "static_distractor",
            "mass_kg":mass if movable else 0.,"rgba":COLORS[color],"color_name":color,
            "semantic_description":description or oid.replace("_"," "),
            **({"parts":parts} if parts else {})}


def part(center, half):
    return {"center":list(center),"half_size":list(half)}


def sphere(oid, position, radius, color, mass=.025):
    return {**box(oid,position,[radius]*3,color,movable=True,mass=mass),"shape":"sphere","radius":radius}


def joint(oid, pivot, axis, mode, limits, parts, color, *, mass=.04, spring=0., friction=.01, damping=.01):
    return {"id":oid,"pivot_world":pivot,"axis_world":axis,"joint_type":mode,"limits":limits,
            "parts":parts,"rgba":COLORS[color],"mass_kg":mass,"spring_N_or_Nm":spring,
            "friction":friction,"damping":damping,"initial_q":0.,"rest_q":0.,
            "semantic_description":oid.replace("_"," "),
            "actuation":"passive native joint; only robot contact may change it during execution"}


def scene(kind, seed=2026102400):
    require(kind in KINDS,"unknown_novel_scene")
    index=KINDS.index(kind)
    s={"schema":"resimpair_novel_scene_v1","id":kind,"seed":seed+index,"table_top_z":0.,
       "robot_roots":[[-.60,-.35,0.],[-.60,.35,0.]],"cameras":deepcopy(CAMERAS),
       "objects":[],"supports":[],"joints":[],"appearance":{
          "table_rgb":[.42+.025*(index%3),.43+.025*(index%4),.44+.02*(index%5)],
          "background_rgb":[.66+.02*(index%4),.71,.74-.025*(index%3)],
          "ambient_rgb":[.38]*3,"key_light_rgb":[1.5]*3},
       "mechanism":NOVELTY[kind][0],"novelty_vs_previous":NOVELTY[kind][1],
       "source_status":"designed_mechanism_probe_without_verified_real_video_window",
       "physics_scope":"rigid bodies and passive joints only; no force injection or actor teleport after reset",
       "asset_origin":"procedural functional geometry, explicit dimensions/mass/contact interfaces"}
    o,j=s["objects"],s["joints"]
    if kind in {"spring_console","rocker_console","linear_faders"}:
        for k,y in enumerate([-.25,.20]):
            color=["red","blue"][k]
            o.append(box("housing_"+str(k),[-.12,y,.022],[.13,.085,.022],"dark"))
            if kind=="spring_console":
                j.append(joint("plunger_"+str(k),[-.15,y,.09],[0.,0.,-1.],"prismatic",[0.,.025],
                               [part([0,0,0],[.039,.039,.012])],color,mass=.01,spring=160.,damping=1.2))
            elif kind=="rocker_console":
                j.append(joint("rocker_"+str(k),[-.13,y,.09],[0.,1.,0.],"revolute",[-.55,.55],
                               [part([0,0,0],[.074,.027,.01])],color,mass=.025,friction=.014,damping=.02))
            else:
                for side in [-1,1]:o.append(box(f"rail_{k}_{side}",[-.10,y+side*.045,.063],[.12,.007,.020],"gray"))
                j.append(joint("fader_"+str(k),[-.18,y,.075],[1.,0.,0.],"prismatic",[0.,.105],
                               [part([0,0,0],[.025,.030,.018])],color,mass=.035,friction=.15,damping=.4))
    elif kind=="rotary_indexer":
        o.append(box("dial_pedestal",[-.13,-.20,.028],[.115,.115,.028],"dark"))
        parts=[part([0,0,0],[.07,.014,.01]),part([0,0,0],[.014,.07,.01]),part([.065,0,.022],[.018,.018,.022])]
        j.append(joint("indexed_dial",[-.13,-.20,.075],[0.,0.,1.],"revolute",[-2.,2.],parts,"yellow",mass=.05,friction=.006,damping=.012))
        for k,a in enumerate([-.9,0,.9]):o.append(box("index_mark_"+str(k),[-.13+.103*math.cos(a),-.20+.103*math.sin(a),.059],[.008,.008,.003],["blue","white","red"][k]))
    elif kind=="balance_beam":
        o.append(box("fulcrum",[-.12,-.18,.055],[.028,.055,.055],"dark"))
        parts=[part([0,0,0],[.125,.035,.008]),part([-.105,0,.022],[.021,.032,.022]),part([.105,0,.022],[.021,.032,.022])]
        j.append(joint("balanced_beam",[-.12,-.18,.13],[0.,1.,0.],"revolute",[-.40,.40],parts,"wood",mass=.12,friction=.012,damping=.025))
        for k,x in enumerate([-.26,.02]):o.append(box("beam_stop_"+str(k),[x,-.18,.045],[.018,.05,.045],"gray"))
    elif kind=="rolling_lanes":
        for k,y in enumerate([-.25,.17]):
            o.append(sphere("ball_"+str(k),[-.22,y,.0355],.035,["red","blue"][k],.035))
            for side in [-1,1]:o.append(box(f"lane_{k}_{side}",[0,y+side*.057,.022],[.31,.006,.022],"wood"))
            o.append(box("end_stop_"+str(k),[.265,y,.04],[.012,.058,.04],"gray"))
            o.append(box("lane_target_"+str(k),[.205,y,.0006],[.048,.050,.0003],"green"))
    elif kind=="hook_retrieval":
        parts=[part([0,0,.023],[.075,.012,.023]),part([.13,0,0],[.075,.010,.014]),part([.215,.045,0],[.012,.055,.014])]
        tool=box("hook",[-.22,-.45,.0145],[.23,.10,.014],"gray",movable=True,mass=.07,parts=parts,
                 description="rigid L hook with a narrow handle and transverse pulling head")
        tool["grasp_feature"]={"center_local":[-.035,0,.025],"closing_local":[0,1,0]}
        o.append(tool)
        for k,y in enumerate([-.23,.10]):
            o.append(box("cargo_"+str(k),[.045,y,.028],[.027,.027,.027],["red","blue"][k],movable=True,mass=.05))
            o.append(box("cargo_destination_"+str(k),[-.13,y,.0006],[.055,.045,.0003],"green"))
    elif kind=="granule_sweeping":
        tool=box("paddle",[-.22,-.45,.019],[.17,.082,.018],"wood",movable=True,mass=.065,
                 parts=[part([0,0,.026],[.075,.013,.025]),part([.13,0,0],[.018,.078,.018])])
        tool["grasp_feature"]={"center_local":[-.025,0,.026],"closing_local":[0,1,0]};o.append(tool)
        o.append(box("handle_rest",[-.245,-.45,.009],[.035,.025,.009],"dark"))
        for k,y in enumerate([-.23,.12]):
            for i in range(3):o.append(sphere(f"grain_{k}_{i}",[-.025+(i%2)*.028,y+(i-1)*.032,.0125],.012,["yellow","blue"][k],.008))
            o.append(box("collector_"+str(k),[.20,y,.0006],[.075,.10,.0003],"green"))
            o.append(box("collector_wall_"+str(k),[.285,y,.035],[.008,.105,.035],"wood"))
    elif kind=="contact_drafting":
        o.append(box("drawing_board",[-.12,-.20,.012],[.15,.14,.012],"white"))
        tool=box("stylus",[-.25,-.45,.0575],[.014,.014,.055],"blue",movable=True,mass=.025,
                 parts=[part([0,0,0],[.007,.007,.055]),part([0,0,.027],[.014,.014,.018])])
        tool["grasp_feature"]={"center_local":[0,0,.029],"closing_local":[0,1,0]};tool["tip_local"]=[0,0,-.055];o.append(tool)
        o.append(box("pen_rest",[-.25,-.45,.001],[.038,.038,.001],"dark"))
    else:
        center=[-.10,-.18,.006]
        parts=[part([0,0,0],[.12,.12,.006])]
        for k in range(12):
            a=2*math.pi*k/12
            parts.append(part([.11*math.cos(a),.11*math.sin(a),.032],[.018,.018,.032]))
        o.append(box("mixing_bowl",center,[.13,.13,.064],"blue",parts=parts))
        tool=box("stirrer",[-.25,-.45,.071],[.020,.018,.07],"wood",movable=True,mass=.035,
                 parts=[part([0,0,.008],[.007,.007,.060]),part([0,0,-.057],[.008,.008,.012]),
                        part([0,0,.035],[.015,.015,.018])])
        tool["grasp_feature"]={"center_local":[0,0,.035],"closing_local":[0,1,0]};tool["tip_local"]=[0,0,-.057];o.append(tool)
        tool["contact_feature"]={"part_index":1,"role":"stir_tip","entry_between":["bead_0","bead_1"],"clearance_margin_m":.002}
        s["tool_free_entry_world"]=[-.10,-.18,.026]
        for i in range(6):
            a=2*math.pi*i/6;o.append(sphere("bead_"+str(i),[-.10+.052*math.cos(a),-.18+.052*math.sin(a),.025],.012,["yellow","red"][i%2],.008))
    # Asset affordances and mass assumptions accompany every scene, not only a
    # visual category label. Selection must not silently replace missing parts.
    from .functional_assets import geometry, preflight
    for item in o:
        item["geometry"]=geometry(item)
        if "grasp_feature" in item:
            width={"hook":.024,"paddle":.026,"stylus":.028,"stirrer":.030}[item["id"]]
            item["grasp_feature"].update(grip_width_m=width,role="tool_handle",approach_local=[0,0,-1])
            item["resting_support"]={"support":"handle_rest" if item["id"]=="paddle" else "pen_rest" if item["id"]=="stylus" else "table",
                "expected_rotation_wxyz":[1,0,0,0],"verification":"native settled pose before approach"}
    preflight(s)
    return s


def task(s, branch, phase="discovery"):
    kind=s["id"];require(branch in {0,1,2},"three_novel_task_branches")
    paired=kind in {"spring_console","rocker_console","linear_faders","rolling_lanes","hook_retrieval","granule_sweeping"}
    selected=[branch] if branch<2 else [0,1]
    params={"selected":selected} if paired else {"direction":1 if branch!=1 else -1,"return_to_origin":branch==2}
    if kind=="contact_drafting":
        params={"points":([[-.21,-.27],[-.03,-.27],[-.12,-.10],[-.21,-.27]] if branch==0 else
                           [[-.21,-.27],[-.03,-.27],[-.03,-.10],[-.21,-.10],[-.21,-.27]] if branch==1 else
                           [[-.21,-.26],[-.15,-.11],[-.09,-.26],[-.03,-.11]]),"height":.024}
    if kind=="bead_stirring":params={"direction":1 if branch!=1 else -1,"turns":2 if branch==2 else 1}
    selected_objects=[]
    if kind=="rolling_lanes":selected_objects=["ball_"+str(i) for i in selected]
    if kind=="hook_retrieval":selected_objects=["hook"]+["cargo_"+str(i) for i in selected]
    if kind=="granule_sweeping":selected_objects=["paddle"]+[f"grain_{k}_{i}" for k in selected for i in range(3)]
    if kind=="contact_drafting":selected_objects=["stylus","drawing_board"]
    if kind=="bead_stirring":selected_objects=["stirrer","mixing_bowl"]+["bead_"+str(i) for i in range(6)]
    if s["joints"]:selected_objects=[s["joints"][i]["id"] for i in (selected if paired else [0])]
    return {"schema":"resimpair_task_intent_v4","id":kind+f"__b{branch}__"+phase,"family":s["mechanism"],
            "scene_kind":kind,"required_skills":[s["mechanism"]],"object_ids":selected_objects,
            "operations":[{"id":"mechanism_program","primitive":s["mechanism"],"parameters":params}],
            "planning_parameters":params,"branch":branch,"goal_description":NOVELTY[kind][1],
            "required_operation_precedence":[],"equivalence_groups":[],"human_hand_policy":"preserve_robot_body_side",
            "non_target_policy":"preserve_identity_and_pose","source_status":s["source_status"],
            "scope":"new mechanism task proposal; source correspondence and execution success must be verified separately"}
