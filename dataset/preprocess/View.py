from PIL import Image
import numpy as np
import math
import torch
import trimesh
from numba import jit
from dataset.preprocess.Object import Obj

ERROR_STR = "Mode is unknown or incompatible with input array shape."


def bytescale(data, c_min=None, c_max=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    c_min : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    c_max : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if c_min is None:
        c_min = data.min()
    if c_max is None:
        c_max = data.max()

    c_scale = c_max - c_min
    if c_scale < 0:
        raise ValueError("`c_max` should be larger than `c_min`.")
    elif c_scale == 0:
        c_scale = 1

    scale = float(high - low) / c_scale
    byte_data = (data - c_min) * scale + low
    return (byte_data.clip(low, high) + 0.5).astype(np.uint8)


def toImage(arr, high=255, low=0, c_min=None, c_max=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            byte_data = bytescale(data, high=high, low=low,
                                  c_min=c_min, c_max=c_max)
            image = Image.frombytes('L', shape, byte_data.tobytes())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            byte_data = (data > high)
            image = Image.frombytes('1', shape, byte_data.tostring())
            return image
        if c_min is None:
            c_min = np.amin(np.ravel(data))
        if c_max is None:
            c_max = np.amax(np.ravel(data))
        data = (data * 1.0 - c_min) * (high - low) / (c_max - c_min) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(ERROR_STR)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in data cube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if 3 in shape:
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    num_channel = shape[ca]
    if num_channel not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    byte_data = bytescale(data, high=high, low=low, c_min=c_min, c_max=c_max)
    if ca == 2:
        str_data = byte_data.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        str_data = np.transpose(byte_data, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        str_data = np.transpose(byte_data, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if num_channel == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(ERROR_STR)

    if mode in ['RGB', 'YCbCr']:
        if num_channel != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if num_channel != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, str_data)
    return image


class TopDownView:
    """
        Take a room, pre-render top-down views
        Of floor, walls and individual objects
        That can be used to generate the multi-channel views used in our pipeline
    """

    def __init__(self, height_cap=4.05, length_cap=6.05, size=512):
        """
        :param height_cap: the maximum height (in meters) of rooms allowed
        :param length_cap: the maximum length/width of rooms allowed.
        :param size: size of the rendered top-down image
        """
        self.size = size
        self.project_generator = ProjectionGenerator(room_size_cap=(length_cap, height_cap, length_cap),
                                                     z_pad=0.5, img_size=size)

    def render(self, room):
        projection = self.project_generator.get_projection(room)

        visualization = np.zeros((self.size, self.size))
        nodes = []

        for node in room.nodes:
            modelId = node.modelId

            t = np.asarray(node.transform).reshape(4, 4)
            # print(t)
            # print(dir(node))

            o = Obj(modelId)

            t = projection.to_2d(t)
            o.transform(t)

            # save_t = t
            # t = projection.to_2d()
            bbox_min = np.dot(np.asarray([node.x_min, node.z_min, node.y_min, 1]), t)
            bbox_max = np.dot(np.asarray([node.x_max, node.z_max, node.y_max, 1]), t)
            x_min = math.floor(bbox_min[0])
            y_min = math.floor(bbox_min[2])
            x_size = math.ceil(bbox_max[0]) - x_min + 1
            y_size = math.ceil(bbox_max[2]) - y_min + 1

            # print(bbox_min, bbox_max)

            description = {
                "modelId": modelId,
                "transform": node.transform,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }

            if y_min < 0:
                y_min = 0
            if x_min < 0:
                x_min = 0

            render = self.render_object(o, x_min, y_min, x_size, y_size, self.size)
            # print(x_min, y_min)

            # print(render.shape)
            description["height_map"] = torch.from_numpy(render).float()

            tmp = self.render_object(o, 0, 0, self.size, self.size, self.size)
            visualization += tmp
            nodes.append(description)

        # print(nodes)
        # Render the floor
        o = Obj(room.modelId + "f", room.house_id, is_room=True)

        t = projection.to_2d()
        o.transform(t)

        a = trimesh.Trimesh(o.vertices, o.faces)
        # a.export('floor.obj')

        floor = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += floor
        floor = torch.from_numpy(floor).float()

        # Render the walls
        o = Obj(room.modelId + "w", room.house_id, is_room=True)

        t = projection.to_2d()
        o.transform(t)
        wall = self.render_object(o, 0, 0, self.size, self.size, self.size)
        visualization += wall
        wall = torch.from_numpy(wall).float()

        return visualization, (floor, wall, nodes)

    @staticmethod
    def render_object(o, x_min, y_min, x_size, y_size, img_size):
        """
        Render a cropped top-down view of object
        :param o: object to be rendered, represented as a triangle mesh
        :param x_min:
        :param y_min:
        :param x_size:
        :param y_size:
        :param img_size:
        :return:
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_helper(triangles, x_min, y_min, x_size, y_size, img_size)

    @staticmethod
    @jit(nopython=True)
    def render_object_helper(triangles, x_min, y_min, x_size, y_size, img_size):
        result = np.zeros((img_size, img_size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0, z0, y0 = triangles[triangle][0]
            x1, z1, y1 = triangles[triangle][1]
            x2, z2, y2 = triangles[triangle][2]
            a = -y1 * x2 + y0 * (-x1 + x2) + x0 * (y1 - y2) + x1 * y2
            if a != 0:
                for i in range(max(0, math.floor(min(x0, x1, x2))),
                               min(img_size, math.ceil(max(x0, x1, x2)))):
                    for j in range(max(0, math.floor(min(y0, y1, y2))),
                                   min(img_size, math.ceil(max(y0, y1, y2)))):
                        x = i + 0.5
                        y = j + 0.5
                        s = (y0 * x2 - x0 * y2 + (y2 - y0) * x + (x0 - x2) * y) / a
                        t = (x0 * y1 - y0 * x1 + (y0 - y1) * x + (x1 - x0) * y) / a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 * (1 - s - t) + z1 * s + z2 * t
                            result[i][j] = max(result[i][j], height)

        return result[min(x_min, x_min + x_size):max(x_min, x_min + x_size),
               min(y_min, y_min + y_size):max(y_min, y_min + y_size)]

    @staticmethod
    @jit(nopython=True)
    def render_object_full_size_helper(triangles, size):
        result = np.zeros((size, size), dtype=np.float32)
        N, _, _ = triangles.shape

        for triangle in range(N):
            x0, z0, y0 = triangles[triangle][0]
            x1, z1, y1 = triangles[triangle][1]
            x2, z2, y2 = triangles[triangle][2]
            a = -y1 * x2 + y0 * (-x1 + x2) + x0 * (y1 - y2) + x1 * y2
            if a != 0:
                for i in range(max(0, math.floor(min(x0, x1, x2))), min(size, math.ceil(max(x0, x1, x2)))):
                    for j in range(max(0, math.floor(min(y0, y1, y2))),
                                   min(size, math.ceil(max(y0, y1, y2)))):
                        x = i + 0.5
                        y = j + 0.5
                        s = (y0 * x2 - x0 * y2 + (y2 - y0) * x + (x0 - x2) * y) / a
                        t = (x0 * y1 - y0 * x1 + (y0 - y1) * x + (x1 - x0) * y) / a
                        if s < 0 and t < 0:
                            s = -s
                            t = -t
                        if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                            height = z0 * (1 - s - t) + z1 * s + z2 * t
                            result[i][j] = max(result[i][j], height)

        return result

    @staticmethod
    def render_object_full_size(o, size):
        """
        Render a full-sized top-down view of the object, see render_object
        """
        triangles = np.asarray(list(o.get_triangles()), dtype=np.float32)
        return TopDownView.render_object_full_size_helper(triangles, size)


class ProjectionGenerator:
    def __init__(self, room_size_cap=(6.05, 4.05, 6.05), z_pad=0.5, img_size=512):
        """
        :param room_size_cap:
        :param z_pad:
        :param img_size:
        """
        self.room_size_cap = room_size_cap
        self.z_pad = z_pad
        self.img_size = img_size
        self.x_scale = self.img_size / self.room_size_cap[0]
        self.y_scale = self.img_size / self.room_size_cap[2]
        self.z_scale = 1.0 / (self.room_size_cap[1] + self.z_pad)

    def get_projection(self, room):
        """
        Generates projection matrices specific to a room,
        need to be room-specific since every room is located in a different position,
        but they are all rendered centered in the image
        :param room:
        :return:
        """
        x_scale, y_scale, z_scale = self.x_scale, self.y_scale, self.z_scale
        # print(x_scale, y_scale, z_scale)
        # print(dir(room))

        x_shift = -(room.x_min * 0.5 + room.x_max * 0.5 - self.room_size_cap[0] / 2.0)
        y_shift = -(room.y_min * 0.5 + room.y_max * 0.5 - self.room_size_cap[2] / 2.0)
        z_shift = -room.z_min + self.z_pad
        # x_shift = -(room.min * 0.5 + room.xmax * 0.5 - self.room_size_cap[0] / 2.0)
        # y_shift = -(room.ymin * 0.5 + room.ymax * 0.5 - self.room_size_cap[2] / 2.0)
        # z_shift = -room.zmin + self.z_pad

        # print(x_shift, y_shift, z_shift)

        t_scale = np.asarray([[x_scale, 0, 0, 0],
                              [0, z_scale, 0, 0],
                              [0, 0, y_scale, 0],
                              [0, 0, 0, 1]])

        t_shift = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [x_shift, z_shift, y_shift, 1]])
        t_3To2 = np.dot(t_shift, t_scale)
        # print(t_3To2)
        # print()

        t_scale = np.asarray([[1 / x_scale, 0, 0, 0],
                              [0, 1 / z_scale, 0, 0],
                              [0, 0, 1 / y_scale, 0],
                              [0, 0, 0, 1]])

        t_shift = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [-x_shift, -z_shift, -y_shift, 1]])

        t_2To3 = np.dot(t_scale, t_shift)

        return Projection(t_3To2, t_2To3, self.img_size)


class Projection:
    def __init__(self, t_2d, t_3d, img_size):
        self.t_2d = t_2d
        self.t_3d = t_3d
        self.img_size = img_size

    def to_2d(self, t=None):
        """
        Parameters
        ----------
        t(Matrix or None): transformation matrix of the object
            if None, then returns room projection
        """
        if t is None:
            return self.t_2d
        else:
            return np.dot(t, self.t_2d)

    def to_3d(self, t=None):
        if t is None:
            return self.t_3d
        else:
            return np.dot(t, self.t_3d)

    def get_orthogonal_parameters(self):
        bottom_left = np.asarray([0, 0, 0, 1])
        top_right = np.asarray([self.img_size, 1, self.img_size, 1])
        bottom_left = np.dot(bottom_left, self.t_3d)
        top_right = np.dot(top_right, self.t_3d)
        return bottom_left[0], top_right[0], bottom_left[2], top_right[2], bottom_left[1], top_right[1]


if __name__ == '__main__':
    from dataset.preprocess.House import House

    h = House(id_="0a9c667d-033d-448c-b17c-dc55e6d3c386")
    # h = House(id_="51515da17cd4b575775cea4f554737a")
    r = h.rooms[1]
    renderer = TopDownView()
    # print(renderer.render(r))
    img = renderer.render(r)[0]
    img = toImage(img, c_min=0, c_max=1)
    img.save("../test.jpg")
