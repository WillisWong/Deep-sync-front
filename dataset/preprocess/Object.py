try:
    # python >= 3.9
    import pickle
except ImportError:
    # python < 3.9
    import pickle5 as pickle
import os
import platform
import numpy as np
import csv

from alive_progress import alive_bar
from Setting import object_data_path, processedData, dataRoot


def parse_objects():
    """
    parse .obj objects and save them to pickle files
    :return:
    """
    obj_dir = object_data_path
    dest = f"{processedData}/object/"
    print("Parsing 3d_front object files...")
    # print(os.listdir(obj_dir))
    obj_nums = len(os.listdir(obj_dir))
    with alive_bar(obj_nums) as bar:
        for (index, model_id) in enumerate(os.listdir(obj_dir)):
            # print("Parsing 3d_front object files...")
            # print(f"{index + 1} of {obj_nums}...")
            if os.path.exists(f"{dest}/{model_id}"):
                bar()
                continue
            if model_id not in ["mgcube", ".DS_Store"]:
                o = Obj(model_id, from_source=True)
                o.save(dest)
                o = Obj(model_id, from_source=True, mirror=True)
                o.save(dest)
                bar()


class ObjectData:
    def __init__(self):
        self.model_to_data = {}

        model_data_file = f"{dataRoot}/Models.csv"
        with open(model_data_file, "r") as f:
            data = csv.reader(f)
            for item in data:
                if item[0] != 'id':  # skip header row
                    self.model_to_data[item[0]] = item[1:]
        # print(self.model_to_data)

    def get_front(self, model_id):
        model_id = model_id.replace("_mirror", "")
        # TODO compensate for mirror (can have effect if not axis-aligned in model space)
        return [float(a) for a in self.model_to_data[model_id][0].split(",")]

    def get_aligned_dims(self, model_id):
        """
        返回模型的规范对齐尺寸
        :param model_id:
        :return:
        """
        # NOTE dims don't change since mirroring is symmetric on yz plane
        model_id = model_id.replace('_mirror', '')
        temp = [item for item in self.model_to_data[model_id][4].split(',')]
        if len(temp) == 4:
            temp = temp[1:]
        return [float(t) / 100.0 for t in temp]

    def get_alignment_matrix(self, model_id):
        """
        Since some models in the dataset are not aligned in the way we want
        Generate matrix that realign them
        :param model_id:
        :return:
        """
        if self.get_front(model_id) == [0, 0, 1]:
            return None
        else:
            # Let's just do case by case enumeration!!!
            if model_id in ["106", "114", "142", "323", "333", "363", "364",
                            "s__1782", "s__1904"]:
                M = [[-1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]]
            elif model_id in ["s__1252", "s__400", "s__885"]:
                M = [[0, 0, -1, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]]
            elif model_id in ["146", "190", "s__404", "s__406"]:
                M = [[0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [-1, 0, 0, 0],
                     [0, 0, 0, 1]]
            else:
                print(model_id)
                return None
                # raise NotImplementedError

            return np.asarray(M)

    def get_model_semantic_frame_matrix(self, model_id):
        """Return canonical semantic frame matrix for model.
           Transforms from semantic frame [0,1]^3, [x,y,z] = [right,up,back] to raw model coordinates."""
        up = np.array([0, 1, 0])  # NOTE: up is assumed to always be +Y for SUNCG objects
        front = np.array(self.get_front(model_id))
        has_mirror = '_mirror' in model_id
        model_id = model_id.replace('_mirror', '')
        h_dims = np.array(self.get_aligned_dims(model_id)) * 0.5
        p_min = np.array([float(a) for a in self.model_to_data[model_id][2].split(',')])
        p_max = np.array([float(a) for a in self.model_to_data[model_id][3].split(',')])
        if has_mirror:
            p_max[0] = -p_max[0]
            p_min[0] = -p_min[0]
        model_space_center = (p_max + p_min) * 0.5
        m = np.identity(4)
        m[:3, 0] = np.cross(front, up) * h_dims[0]  # +x = right
        m[:3, 1] = np.array(up) * h_dims[1]  # +y = up
        m[:3, 2] = -front * h_dims[2]  # +z = back = -front
        m[:3, 3] = model_space_center  # origin = center
        return m

    def get_setIds(self, model_id):
        # print(model_id)
        model_id = model_id.replace("_mirror", "")
        return [a for a in self.model_to_data[model_id][5].split(",")]


class Obj:
    """
       Standard vertex-face representation, triangulated
       Order: x, z, y
    """

    def __init__(self, model_id, house_id=None, from_source=False, is_room=False, mirror=False):
        """

        :param model_id: name of the object to be loaded
        :param house_id: If loading a room, specify which house does the room belong to
        :param from_source: If false, loads the pickled version of the object
                            need to call Object.py at least once to create the pickled version.
        :param is_room: does not apply for rooms
        :param mirror: If true, loads the mirror version
        """
        if is_room:
            from_source = True

        self.vertices = []
        self.faces = []

        if from_source:
            if is_room:
                path = f"{dataRoot}/room/{house_id}/{model_id}.obj"
            else:
                path = f"{dataRoot}/object/{model_id}/{model_id}.obj"

            with open(path, "r") as f:
                for line in f:
                    data = line.split()
                    if len(data) > 0:
                        if data[0] == "v":
                            v = np.asarray([float(i) for i in data[1:4]] + [1])
                            self.vertices.append(v)
                        if data[0] == "f":
                            face = [int(i.split("/")[0]) - 1 for i in data[1:]]
                            if len(face) == 4:
                                self.faces.append([face[0], face[1], face[2]])
                                self.faces.append([face[0], face[2], face[3]])
                            elif len(face) == 3:
                                self.faces.append([face[0], face[1], face[2]])
                            else:
                                print(f"Found a face with {len(face)} edges!!!")

            self.vertices = np.asarray(self.vertices)
            data = ObjectData()
            if not is_room and data.get_alignment_matrix(model_id) is not None:
                self.transform(data.get_alignment_matrix(model_id))
        else:
            with open(f"{processedData}/object/{model_id}/vertices.pkl", "rb") as f:
                self.vertices = pickle.load(f)
            with open(f"{processedData}/object/{model_id}/faces.pkl", "rb") as f:
                self.faces = pickle.load(f)

        self.bbox_min = np.min(self.vertices, 0)
        self.bbox_max = np.max(self.vertices, 0)

        if mirror:
            t = np.asarray([[-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            self.transform(t)
            self.modelId = model_id + "_mirror"
        else:
            self.modelId = model_id

    def save(self, destination=None):
        if destination is None:
            dest_dir_pre = f"{processedData}/object/"
        dest_dir_pre = destination

        dest_dir = f"{dest_dir_pre}/{self.modelId}"

        if not os.path.exists(dest_dir_pre):
            os.makedirs(dest_dir_pre)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(f"{dest_dir}/vertices.pkl", "wb") as f:
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
        with open(f"{dest_dir}/faces.pkl", "wb") as f:
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)

    def transform(self, param):
        self.vertices = np.dot(self.vertices, param)

    def get_triangles(self):
        for face in self.faces:
            yield (self.vertices[face[0]][:3],
                   self.vertices[face[1]][:3],
                   self.vertices[face[2]][:3],)


class ObjectCategories:
    """
        Determine which categories does each object belong to
    """

    def __init__(self):
        filename = "ModelCategoryMapping.csv"
        self.model_to_categories = {}
        # if platform.system() == 'Windows':
        #     model_cat_file = f"{dataRoot}\\{filename}"
        # elif platform.system() == 'Linux':
        model_cat_file = f"{dataRoot}/{filename}"

        with open(model_cat_file, "r", encoding='utf-8', errors='ignore') as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3]]

    def get_fine_category(self, model_id):
        model_id = model_id.replace("_mirror", "")
        return self.model_to_categories[model_id][0]

    def get_coarse_category(self, model_id):
        model_id = model_id.replace("_mirror", "")
        return self.model_to_categories[model_id][1]

    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        :param model_id:
        :return:
        """
        model_id = model_id.replace("_mirror", "")
        category = self.model_to_categories[model_id][0]
        return category


if __name__ == '__main__':
    a = ObjectData()
