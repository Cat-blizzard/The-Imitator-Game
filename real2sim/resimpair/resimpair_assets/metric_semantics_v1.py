"""Source-category hierarchy before name hints; unresolved size/role is pending."""
import copy
import re


def normalize(value):return re.sub(r'[_/\-]+',' ',str(value).strip().lower())


def profile(row,config):
    all_profiles={p['name']:p for p in config['profiles']};all_profiles.update(config['extra_profiles'])
    supplied=row.get('source_categories') or []
    if isinstance(supplied,str):supplied=[supplied]
    if not isinstance(supplied,(list,tuple)):supplied=[]
    categories=list(dict.fromkeys(list(supplied)+[row.get('category'),row.get('native_category')]))
    categories=[str(c) for c in categories if c]
    selected=[]
    for c in categories:
        for key,values in config['category_hierarchy'].items():
            if normalize(c) in [normalize(x) for x in values]:selected.append(all_profiles[key])
    name=normalize(row.get('name',''))
    hinted=next((p for p in config['profiles'] if re.search(p['pattern'],name)),None)
    # Large/soft source categories are never converted to a small dynamic asset
    # merely because a generic fallback geometry happened to fit a gripper.
    static=next((p for p in selected if p['role']=='background'),None)
    if static:
        # Explicit small appliance mechanisms may live inside a broad appliance
        # retrieval group; otherwise the broad group's conservative role wins.
        if static['name']=='large_appliance' and hinted and hinted['name'] in {'tabletop_appliance','phone','small_electronics'}:
            choice=hinted;method='source_category_plus_specific_tabletop_subclass'
        else:choice=static;method='source_category_hierarchy_static_role'
    elif hinted and hinted['role']=='background':choice=hinted;method='specific_large_or_soft_name_evidence'
    elif hinted:choice=hinted;method='specific_name_with_source_categories_recorded'
    elif selected:choice=selected[0];method='source_category_hierarchy'
    else:choice=config['extra_profiles']['role_scale_pending'];method='insufficient_role_and_size_evidence'
    result=copy.deepcopy(choice)
    result['classification_evidence']={'source_categories':categories,'method':method,
        'name_profile':hinted['name'] if hinted else None,
        'category_profiles':list(dict.fromkeys(p['name'] for p in selected)),
        'semantic_truth_verified':False}
    return result
