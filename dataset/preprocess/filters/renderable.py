from dataset.preprocess.Datasets import DatasetFilter
import os

from Setting import dataRoot

"""
Rooms renderable under current setting
Filters out rooms that are too large
and rooms that miss floor/wall objs
"""


def renderable_room_filter(height_cap=4, length_cap=6, width_cap=6):
    def room_criteria(room, house):
        if hasattr(room, 'height'):
            if room.height > height_cap: return False
            if room.length > length_cap: return False
            if room.width > width_cap: return False
        if not os.path.isfile(f"{dataRoot}/room/{room.house_id}/{room.modelId}f.obj"):
            return False
        if not os.path.isfile(f"{dataRoot}/room/{room.house_id}/{room.modelId}w.obj"):
            return False
        for node in room.nodes:
            if node.type == "Box":
                return False
        return True

    dataset_f = DatasetFilter(room_filters=[room_criteria])
    return dataset_f
