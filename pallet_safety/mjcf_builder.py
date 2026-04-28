"""PalletConfig → MuJoCo MJCF (XML) string.

Each item becomes a free body (freejoint) so it can shift, slide, or topple
independently. The pallet base is also a free body. The conveyor is a static
plane. Friction is computed from the pallet temperature via `friction.py`.

Wrap-type constraints (shrink/stretch/banded) are emitted as `<equality>` welds
between the pallet and items, with stiffness scaled by wrap type. WrapType.NONE
emits no constraints — items are held only by gravity + contact friction.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .friction import DEFAULT_PAIR, friction_coefficient
from .models import PalletConfig, WrapType

# (sliding, torsional, rolling) friction triple per MuJoCo convention.
_TORSION = 0.005
_ROLLING = 0.0001

# Wrap stiffness in N/m (qualitative scale; tuned in Phase C with real sims).
_WRAP_STIFFNESS = {
    WrapType.NONE: 0.0,
    WrapType.SHRINK: 200.0,
    WrapType.STRETCH: 800.0,
    WrapType.BANDED: 5000.0,
}


def build_mjcf(
    config: PalletConfig,
    surface_pair: tuple[str, str] = DEFAULT_PAIR,
    conveyor_size_m: tuple[float, float] = (10.0, 2.0),
    actuated_conveyor: bool = True,
) -> str:
    """Build the MuJoCo XML model for the given pallet.

    Args:
        config: pallet to model.
        surface_pair: pallet-belt friction surface pair from `friction_table.json`.
        conveyor_size_m: (length, width) of the modeled conveyor section.
        actuated_conveyor: when True, the conveyor is a body with a velocity
            actuator (Phase C+); when False, it's a static plane (Phase B).
    """
    mu_s, _mu_d = friction_coefficient(
        config.body_temp_c, config.seconds_since_temp_change, surface_pair,
    )
    fric = f"{mu_s:.4f} {_TORSION} {_ROLLING}"

    base_l, base_w, base_h = config.base_dims_m
    base_z = base_h / 2.0  # body origin sits at base CoM

    item_bodies: list[str] = []
    for i, item in enumerate(config.items):
        il, iw, ih = item.dims_m
        ipx, ipy, ipz = item.position
        bx, by, bz = ipx, ipy, ipz + ih / 2.0  # body origin at geom center
        item_bodies.append(
            f'    <body name="item_{i}" pos="{bx:.5f} {by:.5f} {bz:.5f}" '
            f'euler="0 0 {item.orientation_deg}">\n'
            f'      <freejoint name="item_{i}_joint"/>\n'
            f'      <geom name="item_{i}_geom" type="box" '
            f'size="{il/2:.5f} {iw/2:.5f} {ih/2:.5f}" '
            f'mass="{item.weight_kg}" friction="{fric}"/>\n'
            f'    </body>'
        )

    equalities: list[str] = []
    stiffness = _WRAP_STIFFNESS[config.wrap]
    if stiffness > 0:
        # Weld each item to pallet base to model wrap tension.
        for i in range(len(config.items)):
            equalities.append(
                f'    <weld body1="pallet_base" body2="item_{i}" '
                f'solref="0.02 1" solimp="0.95 0.99 0.001"/>'
            )

    eq_block = ""
    if equalities:
        eq_block = "  <equality>\n" + "\n".join(equalities) + "\n  </equality>\n"

    items_block = "\n".join(item_bodies)
    notes_comment = f"  <!-- notes: {escape(config.notes)} -->\n" if config.notes else ""
    meta_comment = (
        f"  <!-- meta: surface_pair={surface_pair[0]}/{surface_pair[1]} "
        f"mu_s={mu_s:.4f} -->\n"
    )

    if actuated_conveyor:
        # Conveyor is a body with a slide joint along +X driven by a velocity
        # actuator. Its top surface sits at z = 0 to keep pallet placement consistent.
        conveyor_block = (
            f'    <body name="conveyor_body" pos="0 0 -0.05">\n'
            f'      <joint name="conveyor_slide" type="slide" axis="1 0 0" damping="0.0"/>\n'
            f'      <geom name="conveyor" type="box" size="{conveyor_size_m[0]} {conveyor_size_m[1]} 0.05" '
            f'mass="500" friction="{fric}" material="beltmat"/>\n'
            f'    </body>'
        )
        actuator_block = (
            "  <actuator>\n"
            '    <velocity name="conveyor_motor" joint="conveyor_slide" kv="10000"/>\n'
            "  </actuator>\n"
        )
    else:
        conveyor_block = (
            f'    <geom name="conveyor" type="plane" size="{conveyor_size_m[0]} {conveyor_size_m[1]} 0.1" '
            f'pos="0 0 0" friction="{fric}" material="beltmat"/>'
        )
        actuator_block = ""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="{escape(config.pallet_id)}">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <material name="palletmat" rgba="0.55 0.38 0.22 1"/>
    <material name="itemmat" rgba="0.78 0.65 0.45 1"/>
    <material name="beltmat" rgba="0.25 0.25 0.28 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
{conveyor_block}
    <body name="pallet_base" pos="0 0 {base_z:.5f}">
      <freejoint name="pallet_joint"/>
      <geom name="pallet_geom" type="box" size="{base_l/2:.5f} {base_w/2:.5f} {base_h/2:.5f}" mass="{config.base_mass_kg}" friction="{fric}" material="palletmat"/>
    </body>
{items_block}
  </worldbody>
{eq_block}{actuator_block}{meta_comment}{notes_comment}</mujoco>
"""


def load_model(config: PalletConfig, **kwargs):
    """Build MJCF and return a loaded MuJoCo MjModel."""
    import mujoco
    xml = build_mjcf(config, **kwargs)
    return mujoco.MjModel.from_xml_string(xml)
