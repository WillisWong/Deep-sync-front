import argparse
import os
import sys

import mathutils
import bpy
import pdb

# add current path to env
sys.path.append(os.getcwd())
print(os.getcwd())
from Setting import json_to_object_save_path, house_data_path


def render_scene(scene_dir):
    room_list = os.listdir(scene_dir)
    for room in room_list:
        file_list = os.listdir(f"{scene_dir}/{room}")
        for file in file_list:
            if file.endswith(".obj"):
                bpy.ops.import_scene.obj(
                    filepath=f"{scene_dir}/{room}/{file}"
                )
        break


def render_blender(model_dir):
    if not os.path.exists(model_dir):
        print(f"{model_dir} is not exist !")
    house_list = os.listdir(model_dir)
    for house_path in house_list:
        render_scene(f"{model_dir}/{house_path}")
        break

    bpy.context.scene.render.engine = 'CYCLES'
    for obj in bpy.context.scene.objects:
        if obj.name in ['Camera']:
            obj.select_set(state=False)
        else:
            obj.select_set(state=False)
            obj.cycles_visibility.shadow = False

    bpy.data.worlds['World'].use_nodes = True
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value[0:3] = (0.75, 0.75, 0.75)

    scene = bpy.context.scene
    bpy.context.scene.cycles.samples = 20
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    # scene.render.alpha_mode = 'TRANSPARENT'
    cam = scene.objects['Camera']
    cam.location = (0, 3.2, 0.8)  # modified
    cam.data.angle = 0.9799147248268127
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'


if __name__ == '__main__':
    # ==========================================================
    # Parse argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--color_depth', type=str, default='8',
                        help='Number of bit per channel used for output. Either 8 or 16.')
    parser.add_argument('--format', type=str, default='PNG',
                        help='Format of files generated. Either PNG or OPEN_EXR')
    parser.add_argument('--scene_path', type=str, default=json_to_object_save_path,
                        help='the path of scene obj files')
    parser.add_argument('--json_path', type=str, default=house_data_path,
                        help='the path of scene json files')
    parser.add_argument('--res', type=float, default=512,
                        help='render image resolution.')
    parser.add_argument('--scale', type=float, default=1,
                        help='Scaling factor applied to model. Depends on size of mesh.')
    parser.add_argument('--remove_doubles', type=bool, default=True,
                        help='Remove double vertices to improve mesh quality.')
    parser.add_argument('--edge_split', type=bool, default=True,
                        help='Adds edge split filter.')
    parser.add_argument('--depth_scale', type=float, default=1.0,
                        help='Scaling that is applied to depth. Depends on size of mesh. '
                             'Try out various values until you get a good result. Ignored if format is OPEN_EXR.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    # ==========================================================

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True
    bpy.context.scene.view_layers["View Layer"].use_pass_environment = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        normalize = tree.nodes.new(type="CompositorNodeNormalize")
        links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        links.new(normalize.outputs[0], depth_file_output.inputs[0])

    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    # scale_normal.use_alpha = True
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    # bias_normal.use_alpha = True
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['Env'], albedo_file_output.inputs[0])

    bpy.data.objects['Cube'].select_set(state=True)
    bpy.ops.object.delete()
    bpy.data.objects['Light'].select_set(state=True)
    bpy.ops.object.delete()

    render_blender(args.scene_path)