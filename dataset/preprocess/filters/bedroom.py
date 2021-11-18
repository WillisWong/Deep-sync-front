from dataset.preprocess.House import *
from dataset.preprocess.Datasets import DatasetFilter
from dataset.preprocess.Object import ObjectCategories
from dataset.preprocess.filters.global_category_filter import *

from Setting import processedData

"""
Bedroom filter
"""


def bedroom_filter(version, source):
    with open(f"{processedData}/{source}/coarse_categories_frequency", "r") as f:
        coarse_categories_frequency = ([s[:-1] for s in f.readlines()])
        coarse_categories_frequency = [s.split(" ") for s in coarse_categories_frequency]
        coarse_categories_frequency = dict([(a, int(b)) for (a, b) in coarse_categories_frequency])
    category_map = ObjectCategories()

    if version == "final":
        filtered, rejected, door_window = GlobalCategoryFilter.get_filter()
        with open(f"{processedData}/{source}/final_categories_frequency", "r") as f:
            frequency = ([s[:-1] for s in f.readlines()])
            result = []
            for s in frequency:
                temp = s.split(" ")
                # print(temp)
                a = temp[0]
                for q in temp[1:-1]:
                    a = a + ' ' + q
                # print(a)
                result.append([a, temp[-1]])
            frequency = result
            # print(frequency)
            # cc()
            # frequency = [s.split(" ") for s in frequency]
            frequency = dict([(a, int(b)) for (a, b) in frequency])

        def node_criteria(node, room):
            if node.modelId in category_map.model_to_categories:
                category = category_map.get_final_category(node.modelId)
            else:
                return False
            if category in filtered: return False
            return True

        def room_criteria(room, house):
            node_count = 0
            # Must have one bed
            bed_count = 0
            for node in room.nodes:
                category = category_map.get_final_category(node.modelId)
                if category in rejected:
                    return False
                if category not in door_window:
                    node_count += 1

                    t = np.asarray(node.transform).reshape((4, 4)).transpose()
                    a = t[0][0]
                    b = t[0][2]
                    c = t[2][0]
                    d = t[2][2]

                    # x_scale = (a**2 + c**2)**0.5
                    # y_scale = (b**2 + d**2)**0.5
                    # z_scale = t[1][1]
                    # print(x_scale,y_scale,z_scale)
                    # if not 0.8<x_scale<1.2: #Reject rooms where any object is scaled by too much
                    #     return False
                    # if not 0.8<y_scale<1.2:
                    #     return False
                    # if not 0.8<z_scale<1.2:
                    #     return False

                if "Bed" in category:
                    bed_count += 1

                # if frequency[category] < 500: return False
            if node_count == 0:
                return False
            # if node_count < 5 or node_count > 20: return False
            if bed_count < 1:
                return False

            return True

    else:
        raise NotImplementedError

    dataset_f = DatasetFilter(room_filters=[room_criteria], node_filters=[node_criteria])

    return dataset_f
