import os
import json
import numpy as np
from dataset.preprocess.Object import ObjectData

from Setting import house_data_path, house_stats_path, house_arch_path


class House:
    """
        Represents a House
    """
    object_data = ObjectData()

    def __init__(self, index=0, id_=None, house_json=None, file_dir=None,
                 include_support_information=False, include_arch_information=False):
        """
        Get a set of rooms from the house which satisfies a certain criteria
        :param index: The index of the house among all houses sorted in alphabetical order
            default way of loading a house
        :param id_: If set, then the house with the specified directory name is chosen
        :param house_json: If set, then the specified json object is used directly to initiate the house
        :param file_dir: `
        :param include_support_information:
        :param include_arch_information:
        """
        if house_json is None:
            if file_dir is None:
                house_dir = f"{house_data_path}/"

                if id_ is None:
                    houses = dict(enumerate(os.listdir(house_dir)))
                    self.id = houses[index]
                    self.__dict__ = json.loads(open(house_dir + houses[index] + "/house.json", 'r').read())
                else:
                    self.id = id_
                    self.__dict__ = json.loads(open(house_dir + id_ + "/house.json", 'r').read())
            else:
                self.__dict__ = json.loads(open(file_dir, 'r').read())
        else:
            self.__dict__ = json.loads(open(file_dir, 'r').read())

        self.filters = []
        self.levels = [Level(l, self) for l in self.levels]
        self.rooms = [r for l in self.levels for r in l.rooms]
        self.nodes = [n for l in self.levels for n in l.nodes]
        self.node_dict = {id_: n for l in self.levels for id_, n in l.node_dict.items()}

        if include_support_information:
            house_stats_dir = f"{house_stats_path}/"
            stats = json.loads(open(house_stats_dir + self.id + "/" + self.id + ".stats.json", 'r').read())
            supports = [(s["parent"], s["child"]) for s in stats["relations"]["support"]]
            for parent, child in supports:
                if child not in self.node_dict:
                    print(f'Warning: support relation {supports} involves not present {child} node')
                    continue
                if "f" in parent:
                    self.get_node(child).parent = "Floor"
                elif "c" in parent:
                    self.get_node(child).parent = "Ceiling"
                elif len(parent.split("_")) > 2:
                    self.get_node(child).parent = "Wall"
                else:
                    if parent not in self.node_dict:
                        print(f'Warning: support relation {supports} involves not present {parent} node')
                        continue
                    self.get_node(parent).child.append(self.get_node(child))
                    self.get_node(child).parent = self.get_node(parent)

        if include_arch_information:
            house_arch_dir = house_arch_path
            arch = json.loads(open(house_arch_dir + self.id + '/' + self.id + '.arch.json', 'r').read())
            self.walls = [
                w for w in arch['elements'] if w['type'] == 'Wall'
            ]

    def get_node(self, id_):
        return self.node_dict[id_]

    def get_rooms(self, filters=None):
        if filters is None:
            filters = self.filters
        if not isinstance(filters, list):
            filters = [filters]
        rooms = self.rooms

        for filter_ in filters:
            rooms = [
                room for room in rooms if filter_(room, self)
            ]
        return rooms

    def filter_rooms(self, filters):
        """
        Similar to get_rooms, but overwrites self.node instead of returning a list
        """
        self.rooms = self.get_rooms(filters)

    def trim(self):
        """
        Get rid of some intermediate attributes
        """
        nodes = list(self.node_dict.values())
        if hasattr(self, 'rooms'):
            nodes.extend(self.rooms)
        for n in nodes:
            for attr in ['xform', 'obb', 'frame', 'model2world']:
                if hasattr(n, attr):
                    delattr(n, attr)
        self.nodes = None
        self.walls = None
        for room in self.rooms:
            room.filters = None
        self.levels = None
        self.node_dict = None
        self.filters = None


class Level:
    """
    Represents a floor level in the house
    """

    def __init__(self, data, house):
        # ===============================================================
        # For Grammar check
        # ===============================================================
        self.__dict__ = data
        self.house = house
        invalid_nodes = [n["id"] for n in self.nodes if (not n["valid"]) and "id" in n]
        self.nodes = [Node(n, self) for n in self.nodes if n["valid"]]
        self.node_dict = {n.id: n for n in self.nodes}
        self.nodes = list(self.node_dict.values())  # deduplicate nodes with same id
        self.rooms = [Room(n, ([self.node_dict[i] for i in [f"{self.id}_{j}" for j in list(set(n.nodeIndices))] if
                                i not in invalid_nodes]), self) \
                      for n in self.nodes if n.is_room() and hasattr(n, 'nodeIndices')]


class Room:
    """
        Represents a room in the house
        """

    def __init__(self, room, nodes, level):
        self.__dict__ = room.__dict__
        self.nodes = nodes
        self.filters = []
        self.house_id = level.house.id

    def get_nodes(self, filters=None):
        """
        Get a set of nodes from the room which satisfies a certain criteria

        Parameters
        ----------
        filters (list[node_filter]): node_filter is tuple[Node,Room] which returns
            if the Node should be included

        Returns
        -------
        list[Node]
        """
        if filters is None: filters = self.filters
        if not isinstance(filters, list): filters = [filters]
        nodes = self.nodes
        for filter_ in filters:
            nodes = [node for node in nodes if filter_(node, self)]
        return nodes

    def filter_nodes(self, filters):
        """
        Similar to get_nodes, but overwrites self.node instead of returning a list
        """
        self.nodes = self.get_nodes(filters)


class Node:
    """
    Basic unit of representation. Usually a room or an object
    """
    warning = True

    def __init__(self, data, level):
        # ===============================================================
        # For Grammar check
        self.type = None
        self.id = None
        self.bbox = {}
        # ======================================================================
        self.__dict__ = data
        self.parent = None
        # self.level = level
        self.child = []

        if hasattr(self, 'bbox'):
            (self.x_min, self.z_min, self.y_min) = self.bbox['min']
            (self.x_max, self.z_max, self.y_max) = self.bbox["max"]
            (self.width, self.length) = sorted([self.x_max - self.x_min, self.y_max - self.y_min])
            self.height = self.z_max - self.z_min
        else:
            if self.warning:
                print(f'Warning: node id={self.id} is valid but has no bbox, setting default values')
            (self.x_min, self.z_min, self.y_min) = (0, 0, 0)
            (self.x_max, self.z_max, self.y_max) = (0, 0, 0)

        if hasattr(self, 'transform') and hasattr(self, 'modelId'):
            # transfer list to array
            t = np.asarray(self.transform).reshape(4, 4)

            # Special cases of models
            try:
                alignment_matrix = House.object_data.get_alignment_matrix(self.modelId)
            except:
                return

            if alignment_matrix is not None:
                t = np.dot(np.linalg.inv(alignment_matrix), t)
                self.transform = list(t.flatten())

            # mirror and adjust transform accordingly
            if np.linalg.det(t) < 0:
                t_reflection = np.asarray(
                    [[-1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
                )
                t = np.dot(t_reflection, t)
                self.modelId += "_mirror"
                self.transform = list(t.flatten())

    def is_room(self):
        return self.type == "Room"


if __name__ == '__main__':
    a = House()
