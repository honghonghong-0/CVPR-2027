"""Minimal-pair renderer for the material-vs-light probe (DRAFT, untested).

Run headless (Blender 4.2+ / Principled BSDF v2):
  blender -b -P render_pairs.py -- --out ./renders --shard 0 --nshard 1

One base config per (material, geometry, view); each variant changes exactly ONE factor
one step away from the base. Metadata -> meta.jsonl (pair_id, changed_factor, level).
Luminance/chroma matching is done afterwards in postprocess (not here).
"""
import bpy, json, math, os, sys, itertools

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(k, d):
    return type(d)(argv[argv.index(k) + 1]) if k in argv else d
OUT = arg("--out", "./renders"); SHARD = arg("--shard", 0); NSHARD = arg("--nshard", 1)
SPP = arg("--spp", 128); RES = arg("--res", 512)
KEYP = arg("--keypower", 150.0); DEVICE = arg("--device", "CPU"); THREADS = arg("--threads", 32); LIMIT = arg("--limit", 0)
os.makedirs(OUT, exist_ok=True)

# ---------------- factors ----------------
MATERIALS = {  # name -> (param key, 3 levels, base index)
    "pearl":   ("thin_film", [0.0, 500.0, 1000.0], 1),   # thin-film thickness nm
    "satin":   ("roughness", [0.05, 0.3, 0.7], 1),
    "velvet":  ("sheen",     [0.0, 0.5, 1.0], 1),
    "frosted": ("trans_rough", [0.05, 0.25, 0.5], 1),
    "glaze":   ("coat",      [0.0, 0.5, 1.0], 1),
}
LIGHT = {  # factor -> (levels, base index)
    "size":   ([0.1, 1.0, 3.0], 1),
    "ratio":  ([2.0, 4.0, 8.0], 1),
    "dir":    ([0.0, 25.0, 50.0], 1),
    "kelvin": ([3200.0, 5600.0, 7500.0], 1),
}
GEOMS = ["sphere", "blob", "cloth"]
VIEWS = [0.0, 30.0]
BASE_COLOR = (0.55, 0.36, 0.42, 1.0)  # soft pink, fixed across all

def kelvin_rgb(k):  # Tanner Helland approximation, normalized
    t = k / 100.0
    r = 255 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = [max(0, min(255, x)) / 255.0 for x in (r, g, b)]
    m = max(c); return tuple(x / m for x in c)

# ---------------- scene ----------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.samples = SPP; s.cycles.use_denoising = True
    if DEVICE == "GPU":
        pr = bpy.context.preferences.addons["cycles"].preferences
        pr.compute_device_type = "CUDA"; pr.get_devices()
        for d in pr.devices: d.use = True
        s.cycles.device = "GPU"
    else:
        s.cycles.device = "CPU"; s.render.threads_mode = "FIXED"; s.render.threads = THREADS
    s.render.resolution_x = s.render.resolution_y = RES
    s.view_settings.view_transform = "Standard"  # no filmic tone curve -> linear-ish comparisons
    w = bpy.data.worlds.new("w"); s.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.18, 0.18, 0.18, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.05
    return s

def add_geom(kind):
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1)
    elif kind == "blob":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1)
        o = bpy.context.object; tex = bpy.data.textures.new("n", "CLOUDS"); tex.noise_scale = 0.8
        m = o.modifiers.new("d", "DISPLACE"); m.texture = tex; m.strength = 0.25
    else:  # cloth-like folded plane
        bpy.ops.mesh.primitive_plane_add(size=2.4); o = bpy.context.object
        o.modifiers.new("s", "SUBSURF").levels = 5
        tex = bpy.data.textures.new("w", "WOOD"); tex.wood_type = "BANDNOISE"; tex.noise_scale = 0.6
        m = o.modifiers.new("d", "DISPLACE"); m.texture = tex; m.strength = 0.2
        o.rotation_euler = (math.radians(60), 0, 0)
    o = bpy.context.object; bpy.ops.object.shade_smooth(); return o

def make_mat(name, key, val):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    p = mat.node_tree.nodes["Principled BSDF"]; I = p.inputs
    I["Base Color"].default_value = BASE_COLOR
    I["Roughness"].default_value = 0.35
    if name == "pearl":
        I["Roughness"].default_value = 0.08; I["Specular IOR Level"].default_value = 1.0
        if "Thin Film Thickness" in I: I["Thin Film Thickness"].default_value = val  # Blender 4.2+
    elif name == "satin":
        I["Roughness"].default_value = val; I["Anisotropic"].default_value = 0.6
    elif name == "velvet":
        I["Roughness"].default_value = 1.0; I["Sheen Weight"].default_value = val
        I["Base Color"].default_value = tuple(x * 0.6 for x in BASE_COLOR[:3]) + (1,)
        I["Sheen Roughness"].default_value = float(os.environ.get("SHEENR",0.3))
    elif name == "frosted":
        I["Transmission Weight"].default_value = 1.0; I["Roughness"].default_value = val
    elif name == "glaze":
        I["Roughness"].default_value = 0.9; I["Coat Weight"].default_value = val
        I["Coat Roughness"].default_value = 0.03
    return mat

def add_lights(size, ratio, direction, kelvin, key_power=600.0):
    col = kelvin_rgb(kelvin)
    a = math.radians(direction)
    def area(nm, pos, power, sz):
        bpy.ops.object.light_add(type="AREA", location=pos); L = bpy.context.object
        L.name = nm; L.data.shape = "DISK"; L.data.size = sz; L.data.energy = power
        L.data.color = col
        d = bpy.context.object.constraints.new("TRACK_TO"); d.target = bpy.data.objects["target"]
        d.track_axis = "TRACK_NEGATIVE_Z"; d.up_axis = "UP_Y"
    r = 4.0
    area("key", (r * math.sin(a), -r * math.cos(a), 2.0), key_power, size)
    area("fill", (-r * math.sin(a + 0.6) , -r * math.cos(a + 0.6), 1.0), key_power / ratio, 2.0)

def add_camera(view_deg):
    a = math.radians(view_deg); bpy.ops.object.camera_add(location=(5 * math.sin(a), -5 * math.cos(a), 1.2))
    c = bpy.context.object; bpy.context.scene.camera = c
    t = c.constraints.new("TRACK_TO"); t.target = bpy.data.objects["target"]
    t.track_axis = "TRACK_NEGATIVE_Z"; t.up_axis = "UP_Y"; c.data.lens = 60

# ---------------- job list ----------------
def jobs():
    for mname, (mkey, mlv, mb) in MATERIALS.items():
        for g, v in itertools.product(GEOMS, VIEWS):
            base = {"mat": mname, "mval": mlv[mb], **{f: lv[b] for f, (lv, b) in LIGHT.items()},
                    "geom": g, "view": v}
            pid = f"{mname}_{g}_{int(v)}"
            yield pid, "base", 1, base
            for i, val in enumerate(mlv):
                if i != mb: yield pid, "material", i, {**base, "mval": val}
            for f, (lv, b) in LIGHT.items():
                for i, val in enumerate(lv):
                    if i != b: yield pid, f, i, {**base, f: val}

def render(job, idx):
    pid, factor, level, c = job
    reset()
    bpy.ops.object.empty_add(location=(0, 0, 0)); bpy.context.object.name = "target"
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -1.05))
    fl = bpy.data.materials.new("floor"); fl.use_nodes = True
    fl.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1)
    fl.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    bpy.context.object.data.materials.append(fl)
    o = add_geom(c["geom"]); o.data.materials.append(make_mat(c["mat"], None, c["mval"]))
    add_lights(c["size"], c["ratio"], c["dir"], c["kelvin"], key_power=KEYP); add_camera(c["view"])
    fn = f"{idx:05d}_{pid}_{factor}{level}.png"
    bpy.context.scene.render.filepath = os.path.join(OUT, fn)
    bpy.ops.render.render(write_still=True)
    with open(os.path.join(OUT, f"meta_{SHARD}.jsonl"), "a") as f:
        f.write(json.dumps({"file": fn, "pair_id": pid, "changed": factor, "level": level, **c}) + "\n")

if __name__ == "__main__":
    J = list(jobs()); print(f"[render_pairs] total jobs {len(J)}")
    if LIMIT: J = J[:LIMIT]
    ONLY = arg("--only", "")  # substring filter on pair_id, e.g. "sphere_0"
    if ONLY: J = [j if ONLY in j[0] else None for j in J]
    for i, j in enumerate(J):
        if j is None: continue
        if i % NSHARD == SHARD and not os.path.exists(os.path.join(OUT, f"{i:05d}_{j[0]}_{j[1]}{j[2]}.png")):
            render(j, i)
    sys.exit(0)
    for i, j in enumerate(J):
        if i % NSHARD == SHARD and not os.path.exists(os.path.join(OUT, f"{i:05d}_{j[0]}_{j[1]}{j[2]}.png")):
            render(j, i)
