from .fluid_nexus_real import (
    read_scene_fluid_nexus_real,
    read_scene_fluid_nexus_real_eval,
)
from .scalar_real import read_scene_scalar_real, read_scene_scalar_real_eval


scene_load_type_callbacks = {
    "scalar_real": read_scene_scalar_real,
    "scalar_real_eval": read_scene_scalar_real_eval,
    "fluid_nexus_real": read_scene_fluid_nexus_real,
    "fluid_nexus_real_eval": read_scene_fluid_nexus_real_eval,
}
