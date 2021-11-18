import json
import trimesh
import numpy as np
import math
import os, argparse
import math
import igl
from shutil import copyfile
from Setting import object_data_path, house_data_path, json_to_object_save_path, room_data_path


def split_path(paths):
    filepath, temp_filename = os.path.split(paths)
    filename, extension = os.path.splitext(temp_filename)
    return filepath, filename, extension


def write_obj_with_tex(save_path, vert, face, v_tex, ft_coor, img_path=None):
    filepath2, filename, extension = split_path(save_path)
    with open(save_path, 'w') as fid:
        fid.write('mtllib ' + filename + '.mtl\n')
        fid.write('usemtl a\n')
        for v in vert:
            fid.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vt in v_tex:
            fid.write('vt %f %f\n' % (vt[0], vt[1]))
        face = face + 1
        ft_coor = ft_coor + 1
        for f, ft in zip(face, ft_coor):
            fid.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
    if img_path is not None and os.path.exists(img_path):
        filepath, filename2, extension = split_path(img_path)
        if not os.path.exists(filepath2 + '/' + filename + extension):
            copyfile(img_path, filepath2 + '/' + filename + extension)
        if img_path is not None:
            with open(filepath2 + '/' + filename + '.mtl', 'w') as fid:
                fid.write('newmtl a\n')
                fid.write('map_Kd ' + filename + extension)


def rotation_matrix(a_xis, theta):
    a_xis = np.asarray(a_xis)
    a_xis = a_xis / math.sqrt(np.dot(a_xis, a_xis))
    a = math.cos(theta / 2.0)
    b, c, d = -a_xis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--future_path',
        default=object_data_path,
        help='path to 3D FUTURE'
    )
    parser.add_argument(
        '--json_path',
        default=house_data_path,
        help='path to 3D FRONT'
    )
    parser.add_argument(
        '--room_path',
        default=room_data_path,
        help='path to 3D FRONT'
    )

    parser.add_argument(
        '--save_path',
        default=json_to_object_save_path,
        help='path to save result dir'
    )
    args = parser.parse_args()

    files = os.listdir(args.json_path)
    # print(files)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for m in files:
        file_path = f"{args.json_path}/{m}"
        with open(file_path + '/house.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            room = {}
            obj = {}
            if not os.path.exists(args.save_path + '/' + m):
                os.mkdir(args.save_path + '/' + m)

            for level in data['levels']:
                nodes = level['nodes']
                for node in nodes:
                    if node['valid'] == 1:
                        if node['type'] == 'Room':
                            node["level"] = level["id"]
                            room[node['id']] = node
                            # room.append(node)
                        elif node['type'] == 'Object':
                            node["level"] = level["id"]
                            obj[node['id']] = node
                        else:
                            print(f"[ERROR] Unknown type {node['type']}!")

            # print(room)
            for room_key in room.keys():
                room_id = room[room_key]['instanceid']
                room_save_path = f"{args.save_path}/{m}/{room_id}"
                if not os.path.exists(room_save_path):
                    os.mkdir(room_save_path)

                model_id = room[room_key]["modelId"]
                room_model_path = f"{args.room_path}/{m}"
                if not os.path.exists(room_model_path):
                    print(f"{room_model_path} is not exists !")

                copyfile(f"{room_model_path}/{model_id}f.obj", f"{args.save_path}/{m}/{room_id}/{model_id}f.obj")
                copyfile(f"{room_model_path}/{model_id}w.obj", f"{args.save_path}/{m}/{room_id}/{model_id}w.obj")

                for j in room[room_key]["nodeIndices"]:
                    _obj = obj[f"{room[room_key]['level']}_{j}"]
                    model_id = _obj["modelId"]
                    if os.path.exists(f"{args.future_path}/{model_id}"):
                        v, vt, _, faces, ftc, _ = igl.read_obj(
                            f"{args.future_path}/{model_id}/{model_id}.obj")

                        # TODO: transform obj and relocation

                        write_obj_with_tex(
                            f"{args.save_path}/{m}/{room_id}/{model_id}.obj", v, faces, vt, ftc,
                            f"{args.future_path}/{model_id}/{model_id}.png")
        break
