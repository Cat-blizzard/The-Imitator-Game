"""Explicit, reversible v2 conventions. Joint spaces remain embodiment-specific.

All offsets in a predicted chunk are relative to the query state, NOT integrated
increments. Raw arrays remain unchanged. Never interpret human poses as commands.
"""
from __future__ import annotations
import numpy as np

V2_NAMES = {
    'imitator_panda_joint16': 'panda_delta_joint16_v2',
    'agibot_dual_gripper_future16': 'agibot_delta_joint16_v2',
    'egodex_wrist20': 'human_delta_wrist20_v2',
}
AGI_ORDER = [*range(7), 14, *range(7, 14), 15]
JOINT_COLUMNS = [*range(7), *range(8, 15)]


def specs_v2():
    common = dict(representation='relative_v2', fps=30, time_reference='query_state',
                  chunk_convention='Each target offset is relative to query t; never cumulative integration.')
    return {
        'panda_delta_joint16_v2': dict(**common, action_dim=16, state_dim=18, native_type='panda',
            layout='[L dq(7), L open(1), R dq(7), R open(1)]',
            coordinate_frame='Panda URDF joint axes and zeros; not world/base/camera Cartesian coordinates',
            units='joint rad; openness dimensionless 0=closed,1=open', target_start_offset=0,
            state_layout='[L q(7), L finger positions(2), R q(7), R finger positions(2)]; rad and m',
            supervision='Native commanded joint targets at t...t+H-1 minus measured q(t).'),
        'agibot_delta_joint16_v2': dict(**common, action_dim=16, state_dim=16, native_type='agibot',
            layout='[L dq(7), L open(1), R dq(7), R open(1)]',
            coordinate_frame='AgiBot native joint axes and zeros; NOT interchangeable with Panda joints',
            units='joint rad; openness dimensionless 0=closed,1=open', target_start_offset=1,
            state_layout='[L q(7), L measured aperture(1), R q(7), R measured aperture(1)]; rad and m',
            supervision='Future measured joints at t+1...t+H minus q(t), plus released gripper signal; auxiliary motion labels, not validated commands.'),
        'human_delta_wrist20_v2': dict(**common, action_dim=20, state_dim=20, native_type='human',
            layout='[L dp(3), L dR columns(6), L pinch(1), R dp(3), R dR columns(6), R pinch(1)]',
            coordinate_frame='Camera frozen at query t, optical x right/y down/z forward; spatial dR=R_future R_current^T in that frame',
            units='translation and thumb-index distance m; rotation columns dimensionless', target_start_offset=1,
            state_layout='Current L/R wrist camera xyz(3), orientation columns(6), pinch distance(1)',
            supervision='Estimated future human motion; pinch is a physical distance, NOT an uncalibrated robot openness command.'),
    }


def relative_actions(row, arrays, t, horizon):
    from .data import ego_actions, agibot_motion
    source = row['source']
    if source == 'imitator_sim':
        s = arrays['states'][t].copy()
        a = arrays['actions'][t:t+horizon].copy()
        a[:, :7] -= s[:7]; a[:, 8:15] -= s[9:16]
        a[:, [7, 15]] = (a[:, [7, 15]] + 1.) / 2.
        return a, s, arrays['mask'][t:t+horizon].copy()
    if source == 'agibot':
        a, s, m = agibot_motion(arrays, t, horizon)
        a = a[:, AGI_ORDER]; s = s[AGI_ORDER].copy(); m = m[:, AGI_ORDER].copy()
        a[:, JOINT_COLUMNS] -= s[JOINT_COLUMNS]
        a[:, [7, 15]] = 1. - a[:, [7, 15]]
        s[[7, 15]] /= 1000.  # Published measured aperture: mm -> m.
        return a, s, m
    if source == 'egodex':
        a, s, m = ego_actions(arrays, t, horizon)
        inv = np.linalg.inv(arrays['camera'][t])
        current = inv @ arrays['wrists'][t]
        future = inv @ arrays['wrists'][t+1:t+1+len(a)]
        delta_r = future[..., :3, :3] @ current[..., :3, :3].swapaxes(-1, -2)
        for hand in range(2):
            k = 10*hand
            a[:, k:k+3] -= s[k:k+3]
            a[:, k+3:k+9] = delta_r[:, hand, :, :2].swapaxes(-1, -2).reshape(-1, 6)
        return a, s, m
    raise ValueError(f'No relative action conversion for {source}')


def panda_to_native(relative, raw_state):
    """NumPy or torch arrays; restores absolute pd_joint_pos commands for execution.

    raw_state [...,18]; relative [...,H,16]. Do this BEFORE temporal aggregation.
    """
    a = relative.clone() if hasattr(relative, 'clone') else relative.copy()
    a[..., :7] += raw_state[..., None, :7]
    a[..., 8:15] += raw_state[..., None, 9:16]
    g = a[..., [7, 15]]
    g = g.clamp(0, 1) if hasattr(g, 'clamp') else np.clip(g, 0, 1)
    a[..., [7, 15]] = 2*g - 1
    return a


def agibot_to_native_motion(relative, aligned_state):
    """Restore original ordering/units; this does not certify a hardware controller."""
    a = relative.clone() if hasattr(relative, 'clone') else relative.copy()
    a[..., JOINT_COLUMNS] += aligned_state[..., None, JOINT_COLUMNS]
    a[..., [7, 15]] = 1. - a[..., [7, 15]]
    return a[..., np.argsort(AGI_ORDER).tolist()]
