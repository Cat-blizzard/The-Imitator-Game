"""Streaming physical observations; standard RecordEpisode owns trajectory data."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .trajectory_execution import as_numpy
from .physical_observation import (contact_identities, contact_allowed, physical_contact_points,
    physical_support_transition, task_object_contact_rows, pose_matrix, measured_source_departure)
from .measurements import _rotation_distance


class PhysicalFailure(RuntimeError):
    pass


class ObservedEnvironment:
    """One env.step per paired action, with sampled native contact monitoring.

    Success termination may be continued to verify release/settling; native
    timeout and failure remain effective. No diagnostic substep archive is kept.
    """

    def __init__(self, env, task, checker_provider, policy):
        self.env, self.unwrapped, self.task = env, env.unwrapped, task
        self.checker_provider, self.policy = checker_provider, policy
        self.phase = "preparation"
        self.last_step = None
        self.references = None
        self.physical_released = [False, False]
        self.reference_frozen = [False, False]
        self.reference_z = [float(as_numpy(a.pose.p)[0, 2]) for a in task.targets]
        self.identities, _ = contact_identities(self.unwrapped, self.unwrapped.agent.agents,
            task.target_names, task.targets)
        self.srdf = {}
        for i, agent in enumerate(self.unwrapped.agent.agents):
            tree = ET.parse(Path(agent.urdf_path).with_suffix(".srdf")).getroot()
            self.srdf[i] = {frozenset((node.attrib['link1'], node.attrib['link2']))
                for node in tree.findall('disable_collisions')}
        self.original_hook = self.unwrapped._after_simulation_step
        self.events = []
        self.pending_errors = []
        self.substeps = self.held_substeps = 0
        self.max_slip_m = np.zeros(2)
        self.max_slip_rad = np.zeros(2)

    def __getattr__(self, name):
        return getattr(self.unwrapped, name)

    def install(self):
        self.unwrapped._after_simulation_step = self.record_substep

    def restore(self):
        self.unwrapped._after_simulation_step = self.original_hook

    def record_substep(self):
        self.original_hook()
        self.substeps += 1
        contacts = self.unwrapped.scene.px.get_contacts()
        checker = self.checker_provider()
        geometry = measured_source_departure(checker, refresh=False)
        rows = task_object_contact_rows(contacts, self.identities, tuple(self.physical_released),
            self.phase, self.task)
        source_contact = [False, False]
        for row in rows:
            if row['physical_contact']:
                for i in row['target_indices']:
                    source_contact[i] |= any(side.get('kind') == 'actor'
                        and side.get('actor') == self.task.initial_supports[i].name for side in row['bodies'])
        for i, actor in enumerate(self.task.targets):
            height = float(as_numpy(actor.pose.p)[0, 2])
            if not self.reference_frozen[i]:
                if geometry[i]:
                    self.reference_frozen[i] = True
                else:
                    self.reference_z[i] = min(self.reference_z[i], height)
            self.physical_released[i] = physical_support_transition(self.physical_released[i],
                geometry[i], self.reference_z[i], height, source_contact[i])
        rows = task_object_contact_rows(contacts, self.identities, tuple(self.physical_released),
            self.phase, self.task)
        for row in rows:
            if row['physical_contact'] and row['class'] == 'forbidden':
                self.pending_errors.append({"reason": "forbidden_payload_contact", "bodies": row['bodies']})
        for contact in contacts:
            sides = [self.identities.get(body.entity, {'kind': 'unknown', 'actor': body.entity.name})
                for body in contact.bodies]
            if not any(side.get('kind') == 'robot' for side in sides):
                continue
            points = [{'separation_m': float(p.separation), 'impulse_Ns': np.asarray(p.impulse).tolist()}
                for p in contact.points]
            if physical_contact_points(points) and not contact_allowed(sides, self.phase, self.srdf):
                self.pending_errors.append({"reason": "forbidden_robot_contact", "bodies": sides})
        if self.references is not None and (self.phase == 'lift_and_hold' or self.phase.startswith('carry_')):
            self.held_substeps += 1
            for i, (agent, actor) in enumerate(zip(self.unwrapped.agent.agents, self.task.targets)):
                tcp = pose_matrix(as_numpy(agent.tcp.pose.raw_pose)[0])
                obj = pose_matrix(as_numpy(actor.pose.raw_pose)[0])
                actual = np.linalg.inv(tcp) @ obj
                self.max_slip_m[i] = max(self.max_slip_m[i],
                    np.linalg.norm(actual[:3, 3] - self.references[i, :3, 3]))
                self.max_slip_rad[i] = max(self.max_slip_rad[i],
                    _rotation_distance(self.references[i, :3, :3], actual[:3, :3]))

    def step(self, action):
        if int(as_numpy(self.unwrapped.elapsed_steps).item()) >= self.policy.max_episode_steps:
            raise RuntimeError("Official episode budget exhausted")
        self.pending_errors = []
        self.last_step = self.env.step(action)
        obs, reward, terminated, truncated, info = self.last_step
        if self.pending_errors:
            self.events.extend({"phase": self.phase, "step": int(as_numpy(self.unwrapped.elapsed_steps).item()),
                **event} for event in self.pending_errors)
            raise PhysicalFailure(self.pending_errors[0]['reason'])
        success = bool(as_numpy(info.get('success', False)).item())
        fail = bool(as_numpy(info.get('fail', False)).item())
        continued = bool(as_numpy(terminated).item()) and success and not fail and not bool(as_numpy(truncated).item())
        return obs, reward, False if continued else terminated, truncated, info

    def summary(self):
        return {"substeps": self.substeps, "held_substeps": self.held_substeps,
            "forbidden_contacts": self.events, "physical_source_departure": self.physical_released,
            "max_held_position_slip_m": self.max_slip_m.tolist(),
            "max_held_rotation_slip_rad": self.max_slip_rad.tolist()}
