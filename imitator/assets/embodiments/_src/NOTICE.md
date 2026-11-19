# Vendored embodiment sources

Source robot descriptions used by `imitator.embodiments.registry.build_assets()` and referenced (by relative path)
from the derived URDFs in `imitator/assets/embodiments/<arm>/`. Only the files those URDFs need are kept; the
camera meshes (`camera.glb`, links dropped by the derivation) are omitted.

| folder | source | license |
|---|---|---|
| `robotwin/ARX-X5`, `robotwin/piper`, `robotwin/ur5-wsg` | RoboTwin 2.0 embodiment assets (`assets/embodiments/`), ARX X5, AgileX PiPER, UR5 + Weiss WSG-50 | MIT (`robotwin/LICENSE`) |
| `xarm6` | ManiSkill-XArm6 v0.1.1 (`https://github.com/haosulab/ManiSkill-XArm6`), xArm6 + Robotiq 2F-85 / Allegro | see `xarm6/LICENSE` |
| `allegro` | Wonik Allegro Hand v4 as shipped in ManiSkill 3.0.1 (`mani_skill/assets/robots/allegro`) | see `allegro/LICENSE` |
