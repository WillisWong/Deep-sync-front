from dataset.preprocess.Datasets import Dataset, DatasetFilter, DatasetSaver, DatasetRender, DataToJSON, DatasetStats

# ==================EDIT HERE====================
filter_description = [("bedroom", "final"), ("collision",)]
source = "bedroom"  # Source and dest are relative to utils.get_data_root_dir()
destination = "bedroom_fin"
# I hate typing True and False
stats = 1  # If true, print stats about the dataset
save_freq = 1  # If true, save category frequency count to the dest directory
render = 1  # If true, render the dataset
save = 0  # If true, save the filtered dataset back as pkl files
json = 1  # If true, save the json files
# There seems to be a KNOWN BUG (I can't remember) that prevents using render/json together with save
# So avoid using them together to be save
render_size = 512


# ===============================================


def filter_process(filter_description, source, destination, stats, save_freq, render, save, is_json, render_size=512):
    actions = []
    for description in filter_description:
        actions.append(get_filter(source, *description))
    if stats:
        actions.append(DatasetStats(save_freq=save_freq, details=False, model_details=False, save_dest=destination))
    if save:
        actions.append(DatasetSaver(destination=destination, batch_size=1000))
    if render:
        actions.append(DatasetRender(destination=destination, size=render_size))
    if is_json:
        actions.append(DataToJSON(destination=destination))

    d = Dataset(actions, source=source)
    d.run()


def get_filter(source, filter, *args):
    dataset_f = None
    if filter == "good_house":
        from dataset.preprocess.filters import good_house_criteria
        house_f = good_house_criteria
        dataset_f = DatasetFilter(house_filters=[house_f])
    elif filter == "room_type":
        from dataset.preprocess.filters import room_type_criteria
        room_f = room_type_criteria(*args)
        dataset_f = DatasetFilter(room_filters=[room_f])
    elif filter == "bedroom":
        from dataset.preprocess.filters import bedroom_filter
        dataset_f = bedroom_filter(*args, source)
    elif filter == "office":
        from dataset.preprocess.filters.office import office_filter
        dataset_f = office_filter(*args, source)
    elif filter == "livingroom":
        from dataset.preprocess.filters.livingroom import livingroom_filter
        dataset_f = livingroom_filter(*args, source)
    elif filter == "floor_node":
        from dataset.preprocess.filters import floor_node_filter
        dataset_f = floor_node_filter(*args)
    elif filter == "renderable":
        from dataset.preprocess.filters.renderable import renderable_room_filter
        dataset_f = renderable_room_filter(*args)
    elif filter == "collision":
        from dataset.preprocess.filters import collision_filter
        from dataset.preprocess.priors.observations import ObjectCollection
        oc = ObjectCollection()
        dataset_f = collision_filter(oc)
    else:
        raise NotImplementedError
    return dataset_f
