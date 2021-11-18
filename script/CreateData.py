import sys
from DataFilter import filter_process
from dataset.preprocess.Datasets import create_dataset
from dataset.preprocess.Object import parse_objects


def init_dataset():
    print("Creating initial dataset...")
    create_dataset(dest='temp')


def create_bedroom_dataset():
    print("Creating bedroom dataset...")
    filter_description = [("room_type", ["Bedroom", "MasterBedroom", "SecondBedroom"]), ("floor_node",),
                          ("renderable",)]

    filter_process(filter_description, "temp", "bed_temp", 1, 1, 0, 1, 0)

    filter_description = [("bedroom", "final")]

    filter_process(filter_description, "bed_temp", "bedroom", 1, 1, 1, 0, 1)

    # data, dest = read_raw_data("bedroom")
    # save_binary_dataset(data, dest)


def create_living_room_dataset():
    print("Creating living room dataset...")
    #
    filter_description = [("room_type", ["LivingRoom", "LivingDiningRoom"]), ("renderable",)]
    filter_process(filter_description, "temp", "livingroom_temp", 1, 1, 0, 1, 0)

    filter_description = [("livingroom", "final")]
    filter_process(filter_description, "livingroom_temp", "livingroom", 1, 1, 1, 0, 1)

    # data, dest = read_raw_data("livingroom")
    # save_binary_dataset(data, dest)


if __name__ == '__main__':
    parse_objects()
    init_dataset()

    create_bedroom_dataset()

    create_living_room_dataset()








