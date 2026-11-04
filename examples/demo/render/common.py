"""3 つのデモで共通の Blender (bpy) ヘルパ。"""
import math

import bpy
import numpy as np
from mathutils import Vector


def reset_scene(res=(1920, 1080), samples=128, engine="CYCLES"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "16"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "NONE"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU" if backend != "NONE" else "CPU"
    return sc


def sky_world(sun_elevation=35.0, sun_rotation=40.0, strength=1.0):
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False  # 太陽は別途 Sun ライトで置く (二重照明を避ける)
    sky.sun_elevation = math.radians(sun_elevation)
    sky.sun_rotation = math.radians(sun_rotation)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])


def camera(location, target, lens=35.0):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = lens
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def material(name, base=(0.8, 0.8, 0.8, 1), rough=0.5, **inputs):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base
    b.inputs["Roughness"].default_value = rough
    for k, v in inputs.items():
        b.inputs[k.replace("_", " ")].default_value = v
    return m


def mesh_object(name, verts, faces, mat=None, smooth=True):
    """numpy の頂点/面から mesh オブジェクトを作る (Y-up のシミュ座標 -> Blender の Z-up へ変換)。"""
    v = np.asarray(verts, np.float32)[:, [0, 2, 1]] * np.array([1, -1, 1], np.float32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.ravel())
    f = np.asarray(faces, np.int32)
    me.loops.add(f.size)
    me.polygons.add(len(f))
    me.loops.foreach_set("vertex_index", f.ravel())
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3, dtype=np.int32))
    me.update()
    me.validate()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(f), bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if mat:
        ob.data.materials.append(mat)
    return ob


def to_blender(P):
    """sim_to_blender の numpy 版 (n,3)。"""
    P = np.asarray(P, np.float32)
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)


def sim_to_blender(p):
    """シミュ座標 (x,y,z; Y-up) -> Blender 座標 (x, -z, y)。"""
    return (p[0], -p[2], p[1])


def box(name, center, half, mat=None):
    bpy.ops.mesh.primitive_cube_add(location=sim_to_blender(center))
    ob = bpy.context.object
    ob.name = name
    ob.scale = (half[0], half[2], half[1])
    if mat:
        ob.data.materials.append(mat)
    return ob


def cylinder(name, center, radius, half_height, mat=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=2 * half_height, vertices=64, location=sim_to_blender(center))
    ob = bpy.context.object
    ob.name = name
    if mat:
        ob.data.materials.append(mat)
    return ob


def render_still(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
