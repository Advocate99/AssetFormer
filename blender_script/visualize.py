import os
import json
import bpy
import math
import mathutils


## Edit the paths below to point to the correct directories on your system
src_dir = 'pathtosample/'
asset_dir = 'pathtofbxfiles/'
output_dir = 'pathtooutput/'

files = os.listdir(src_dir)

asset_dict = {
    22001000: 'wall.fbx',
    22002000: 'right_angle_wall.fbx',
    22003000: 'triangular_wall.fbx',
    22009000: 'inverted_right_angle_wall.fbx',
    22011000: 'low_wall.fbx',
    22022000: 'low_right_angle_wall.fbx',
    22101000: 'rectangular_column.fbx',
    22201000: 'beam.fbx',
    21902000: 'floor.fbx',
    22301000: 'floor.fbx',
    22302000: 'sloped_roof.fbx',
    22303000: 'high_sloped_roof.fbx',
    22310000: 'low_sloped_roof.fbx',
    22311000: 'convex_roof.fbx',
    22312000: 'low_convex_roof.fbx',
    22313000: 'concave_roof.fbx',
    22314000: 'low_concave_roof.fbx',
    22401000: 'sloped_ladder.fbx',
    22402000: 'low_sloped_ladder.fbx',
    22405000: 'straight_ladder.fbx',
    22501000: 'door.fbx',
    22505000: 'wide_door.fbx',
    22601000: 'window_wall.fbx',
    22603000: 'balcony.fbx',
    22604000: 'railing.fbx',
}

for i in range(len(files)):
    with open(os.path.join(src_dir, files[i])) as f:
        data = json.load(f)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    elements = data['elements']

    for j, element in enumerate(elements):
        building_id = element['building_id']
        fbx_path = os.path.join(asset_dir, asset_dict[building_id])
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        obj = bpy.context.selected_objects[0]
        bpy.ops.object.select_all(action='DESELECT')
        x = element['location']['x'] * 0.001
        y = element['location']['y'] * 0.001
        z = element['location']['z'] * 0.001
        obj.location = (x, -y, z)
        angle = obj.rotation_euler[2] - math.radians(element['rotation']['yaw'])
        obj.rotation_euler = (obj.rotation_euler[0], obj.rotation_euler[1], angle)
    
    obj = bpy.context.object
    render_path = os.path.join(output_dir, files[i].split('.')[0])
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.opengl(write_still=True)