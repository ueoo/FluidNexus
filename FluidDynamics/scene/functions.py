import numpy as np

from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud, get_world_2_view2


def get_nerf_pp_norm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = get_world_2_view2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        # os.makedirs("cam_vis", exist_ok=True)
        # np.save(f"cam_vis/cam_{cam.image_name}_pose.npy", C2W)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetch_ply(path, gray_image=False):
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    times = np.vstack([vertices["t"]]).T if "t" in vertices else None
    if gray_image:
        colors = np.vstack([vertices["gray"]]).T / 255.0
    else:
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def store_ply(path, xyzt, rgb, gray_image=False):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("t", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    if gray_image:
        dtype += [("gray", "u1")]
    else:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def shift_image(image, offset_h, offset_w):
    shifted_image = np.zeros_like(image)

    # Perform the shift
    if offset_h > 0 and offset_w > 0:
        shifted_image[offset_h:, offset_w:, :] = image[:-offset_h, :-offset_w, :]
    elif offset_h > 0 and offset_w < 0:
        shifted_image[offset_h:, :offset_w, :] = image[:-offset_h, -offset_w:, :]
    elif offset_h < 0 and offset_w > 0:
        shifted_image[:offset_h, offset_w:, :] = image[-offset_h:, :-offset_w, :]
    elif offset_h < 0 and offset_w < 0:
        shifted_image[:offset_h, :offset_w, :] = image[-offset_h:, -offset_w:, :]
    elif offset_h > 0 and offset_w == 0:
        shifted_image[offset_h:, :, :] = image[:-offset_h, :, :]
    elif offset_h < 0 and offset_w == 0:
        shifted_image[:offset_h, :, :] = image[-offset_h:, :, :]
    elif offset_h == 0 and offset_w > 0:
        shifted_image[:, offset_w:, :] = image[:, :-offset_w, :]
    elif offset_h == 0 and offset_w < 0:
        shifted_image[:, :offset_w, :] = image[:, -offset_w:, :]

    return shifted_image


def rotate_camera_around_z_axis(C2W, theta):
    # Rotation matrix around Z-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_roll = np.array([[cos_theta, -sin_theta, 0, 0], [sin_theta, cos_theta, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Update camera-to-world matrix
    C2W_new = np.dot(C2W, R_roll)
    return C2W_new


def rotate_camera_around_y_axis(C2W, theta):
    # Rotation matrix around Y-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_pitch = np.array([[cos_theta, 0, sin_theta, 0], [0, 1, 0, 0], [-sin_theta, 0, cos_theta, 0], [0, 0, 0, 1]])

    # Update camera-to-world matrix
    C2W_new = np.dot(C2W, R_pitch)
    return C2W_new


def rotate_camera_around_x_axis(C2W, theta):
    # Rotation matrix around X-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_yaw = np.array([[1, 0, 0, 0], [0, cos_theta, -sin_theta, 0], [0, sin_theta, cos_theta, 0], [0, 0, 0, 1]])

    # Update camera-to-world matrix
    C2W_new = np.dot(C2W, R_yaw)
    return C2W_new


def generate_camera_poses(M0, radius, num_cameras):
    R0 = M0[:3, :3]  # Extract rotation matrix
    t0 = M0[:3, 3]  # Extract translation vector

    # Camera axes
    r0 = R0[:, 0]  # Right vector
    u0 = R0[:, 1]  # Up vector

    camera_poses = []

    for i in range(num_cameras):
        theta = (2 * np.pi * i) / num_cameras
        offset = radius * (np.cos(theta) * r0 + np.sin(theta) * u0)
        ti = t0 + offset

        # Construct new camera-to-world matrix
        Mi = np.eye(4)
        Mi[:3, :3] = R0
        Mi[:3, 3] = ti

        camera_poses.append(Mi)

    return camera_poses
