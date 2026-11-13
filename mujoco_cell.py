"""Optional CPU MuJoCo development cell: pinned iiwa arm, ideal rigid grasp."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from robot import Rejected, example_scene, vector


MODEL_SHA256 = "ee7e6d796a9f8642de6097f95077281d15f43678dc5285beafeda99b84cf9ef5"


def scene_model(path, scene):
    """Retain published collision geometry; remove visual meshes and add a small cell."""
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != MODEL_SHA256:
        raise Rejected("Use the pinned Menagerie iiwa14.xml model documented in README.")
    root = ET.fromstring(source)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"mesh", "keyframe"} or child.get("class") == "visual":
                parent.remove(child)
    root.find("option").set("timestep", "0.002")
    world = root.find("worldbody")
    world.find("body").set("pos", "-0.5 0 -0.25")
    ET.SubElement(world, "geom", name="ground", type="plane", pos="0 0 -0.25", size="1 1 0.1")
    ET.SubElement(world, "geom", name="table", type="box", pos="0.025 0 -0.02",
                  size="0.375 0.35 0.02", rgba="0.5 0.35 0.2 1")
    hand = ET.SubElement(root.find(".//body[@name='link7']"), "body", name="tool")
    ET.SubElement(hand, "geom", name="tool_geom", type="capsule", size="0.008",
                  fromto="0 0 0.055 0 0 0.17", mass="0.1", rgba="0.1 0.1 0.1 1")
    ET.SubElement(hand, "site", name="tcp", pos="0 0 0.18", size="0.004")
    equality = ET.SubElement(root, "equality")
    for key, obj in scene["objects"].items():
        position = " ".join(str(x / 1000) for x in obj["position"])
        size = " ".join(str(x / 2000) for x in obj["size"])
        body = ET.SubElement(world, "body", name=key, pos=position)
        ET.SubElement(body, "freejoint", name="free_" + key)
        shape = "cylinder" if key == "can" else "box"
        if shape == "cylinder":
            size = f"{obj['size'][0] / 2000} {obj['size'][2] / 2000}"
        ET.SubElement(body, "geom", name="object_" + key, type=shape, size=size,
                      mass="0.1", friction="1 0.005 0.0001", rgba="0.7 0.2 0.2 1")
        ET.SubElement(equality, "weld", name="grasp_" + key, body1="tool", body2=key,
                      active="false", solref="0.005 1")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class MuJoCoCell:
    """Dynamics and observed contacts; grasp is an ideal weld, not learned finger control.

    Collision checks sample the whole published collision model, including during
    execution. They are not a proof of continuous collision freedom or real safety.
    """

    backend_name = "MuJoCo"
    version, model_digest = mj.__version__, MODEL_SHA256
    joint_limit_tolerance_rad = .001

    def __init__(self, model_path, scene=None):
        self.description = deepcopy(scene or example_scene())
        if scene is None:
            self.description["tcp"] = [-50, -250, 200]
            self.description["objects"]["red"]["position"] = [-50, -150, 20]
        self.description["bounds"] = [[-350, -350, 0], [400, 350, 600]]
        self.model = scene_model(model_path, self.description)
        self.data = mj.MjData(self.model)
        self.site = self.model.site("tcp").id
        self.carried, self.stopped = None, False
        self.reset_trace()
        self.data.qpos[:7] = [0, .8, 0, -1.6, 0, .8, 0]
        self.data.qpos[:7] = self.ik(self.data, np.array(self.description["tcp"]) / 1000)
        self.data.ctrl[:] = self.data.qpos[:7]
        mj.mj_forward(self.model, self.data)
        self.check_contacts(self.data)
        self.advance(100)
        self.reset_trace()

    def reset_trace(self):
        self.trace, self.releases, self.motions = [], [], 0
        self.samples, self.collision_events, self._trace_start = [], [], None
        self.started_at = float(self.data.time)
        self.max_joint_limit_error_rad = 0.

    def snapshot(self):
        scene = deepcopy(self.description)
        scene["tcp"] = (self.data.site_xpos[self.site] * 1000).tolist()
        for key, obj in scene["objects"].items():
            obj["position"] = (self.data.body(key).xpos * 1000).tolist()
        return scene

    def ik(self, data, point):
        """Damped Jacobian IK, keeping the tool pointed vertically down."""
        desired = np.diag([1., -1., -1.])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(350):
            mj.mj_forward(self.model, data)
            position = point - data.site_xpos[self.site]
            orientation = Rotation.from_matrix(desired @ data.site_xmat[self.site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(position) < .0001 and np.linalg.norm(orientation) < .001:
                return data.qpos[:7].copy()
            mj.mj_jacSite(self.model, data, jacp, jacr, self.site)
            jac = np.vstack((jacp[:, :7], .15 * jacr[:, :7]))
            error = np.r_[position, .15 * orientation]
            change = jac.T @ np.linalg.solve(jac @ jac.T + .00001 * np.eye(6), error)
            data.qpos[:7] = np.clip(data.qpos[:7] + np.clip(change, -.12, .12),
                                   self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        raise Rejected(f"MuJoCo IK could not reach {np.round(point * 1000, 2).tolist()} mm with a downward tool.")

    def check_contacts(self, data, carried=None):
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            bodies = {int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            names = {self.model.body(i).name for i in bodies}
            geoms = {self.model.geom(g).name for g in (contact.geom1, contact.geom2)}
            if geoms & {"ground"} and "base" in names:
                continue
            objects = names & self.description["objects"].keys()
            if "table" in geoms and objects:
                key = next(iter(objects))
                half_height = self.description["objects"][key]["size"][2] / 2000
                if data.body(key).xpos[2] - half_height >= -.001:
                    continue  # Expected support, within the declared 1 mm tolerance.
            if carried and names == {"tool", carried}:
                continue
            raise Rejected("MuJoCo contact: " + ", ".join(sorted(names)))

    def attach_probe(self, data, carried):
        if carried:
            address = int(self.model.joint("free_" + carried).qposadr[0])
            height = self.description["objects"][carried]["size"][2] / 2000 + .01
            data.qpos[address:address + 3] = data.site_xpos[self.site] - [0, 0, height]
            data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
            mj.mj_forward(self.model, data)

    async def preflight(self, plan):
        probe = mj.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        mj.mj_forward(self.model, probe)
        carried = None
        for kind, target in plan["steps"]:
            if kind != "move":
                carried = target if kind == "pick" else None
                continue
            start, end = probe.site_xpos[self.site].copy(), np.array(target) / 1000
            count = max(1, math.ceil(np.linalg.norm(end - start) / .005))
            for t in np.linspace(0, 1, count + 1)[1:]:
                probe.qpos[:7] = self.ik(probe, start + t * (end - start))
                self.attach_probe(probe, carried)
                self.check_contacts(probe, carried)

    def advance(self, count):
        for _ in range(count):
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mj.mj_step(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            limits, angles = self.model.jnt_range[:7], self.data.qpos[:7]
            excess = np.maximum(limits[:, 0] - angles, angles - limits[:, 1]).max()
            self.max_joint_limit_error_rad = max(self.max_joint_limit_error_rad, float(excess))
            try:
                self.check_contacts(self.data, self.carried)
            except Rejected as error:
                self.collision_events.append(dict(time=float(self.data.time), reason=str(error)))
                raise
            if len(self.samples) == 0 or self.data.time - self.samples[-1][0] >= .0099:
                self.samples.append((float(self.data.time), (self.data.site_xpos[self.site] * 1000).tolist()))
                if self.carried:
                    position = self.data.body(self.carried).xpos.copy()
                    self.trace.append((self.carried, (self._trace_start * 1000).tolist(),
                                       (position * 1000).tolist()))
                    self._trace_start = position

    async def move(self, point):
        if self.stopped:
            raise Rejected("MuJoCo cell is stopped.")
        self.motions += 1
        start, end = self.data.site_xpos[self.site].copy(), np.array(point) / 1000
        probe = mj.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        count = max(1, math.ceil(np.linalg.norm(end - start) / .002))
        for t in np.linspace(0, 1, count + 1)[1:]:
            self.data.ctrl[:] = self.ik(probe, start + t * (end - start))
            self.advance(10)  # 2 mm per 20 ms: nominal Cartesian speed 100 mm/s.
        for _ in range(100):
            if np.linalg.norm(self.data.site_xpos[self.site] - end) < .0005:
                return
            self.advance(10)
        raise Rejected("MuJoCo tool did not settle within 0.5 mm.")

    async def grip(self, action, object_id):
        equality = self.model.equality("grasp_" + object_id).id
        if action == "pick":
            obj = self.data.body(object_id)
            expected = obj.xpos + [0, 0, self.description["objects"][object_id]["size"][2] / 2000 + .01]
            if np.linalg.norm(self.data.site_xpos[self.site] - expected) > .001:
                raise Rejected("Tool is not at the configured grasp point.")
            hand = self.data.body("tool")
            rotation = hand.xmat.reshape(3, 3)
            self.model.eq_data[equality, 3:6] = rotation.T @ (obj.xpos - hand.xpos)
            mj.mju_mat2Quat(self.model.eq_data[equality, 6:10], (rotation.T @ obj.xmat.reshape(3, 3)).ravel())
            self.data.eq_active[equality] = 1
            self.carried = object_id
            self._trace_start = obj.xpos.copy()
        else:
            self.data.eq_active[equality] = 0
            self.carried = None
            self.advance(100)
            self.releases.append(object_id)

    def reposition(self, object_id, point):
        """Check a discrete scene edit for contact before changing the live state."""
        if self.carried:
            raise Rejected("Release the payload before editing the scene.")
        point = np.array(vector(point)) / 1000
        joint = self.model.joint("free_" + object_id)
        address, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        probe = mj.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        probe.qpos[address:address + 3] = point
        mj.mj_forward(self.model, probe)
        self.check_contacts(probe)
        self.data.qpos[address:address + 3] = point
        self.data.qvel[dof:dof + 6] = 0
        mj.mj_forward(self.model, self.data)
        self.advance(100)

    def stop(self):
        self.stopped = True
