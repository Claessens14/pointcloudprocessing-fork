import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d
import torch
import colorsys
import random

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        import ipdb; ipdb.set_trace()
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

if __name__ == '__main__':
    # Depth Camera parameters:
    FX_DEPTH = 5.8262448167737955e+02
    FY_DEPTH = 5.8269103270988637e+02
    CX_DEPTH = 3.1304475870804731e+02
    CY_DEPTH = 2.3844389626620386e+02

    # RGB camera intrinsic Parameters:
    FX_RGB = 5.1885790117450188e+02
    FY_RGB = 5.1946961112127485e+02
    CX_RGB = 3.2558244941119034e+0
    CY_RGB = 2.5373616633400465e+02

    # Rotation matrix:
    R = -np.array([[9.9997798940829263e-01, 5.0518419386157446e-03, 4.3011152014118693e-03],
                   [-5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03],
                   [- 4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01]])
    # Translation vector:
    T = np.array([2.5031875059141302e-02, -2.9342312935846411e-04, 6.6238747008330102e-04])

    # Read depth and color image:
   # depth_image = iio.imread('../data/depth.png')
   # rgb_image = iio.imread('../data/rgb.jpg')

    observations = torch.load("../train_observations_lst.pt")
    step_index = 3#8 #3 #8 #15 #8 #3
    end = len(observations[step_index]['depth'])
    rgb_image = observations[step_index]['rgb'][:end]#[:, 80:-80]
    depth_image = observations[step_index]['depth'][:end]#[:, 80:-80]
    semantic_image = observations[step_index]['semantic'][:end]
    #depth_image = iio.imread('../data/depth.png')
    depth_image = depth_image.squeeze()

    # Display depth and grayscale image:
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(depth_image, cmap="gray")
    axs[0].set_title('Depth image')
    axs[1].imshow(rgb_image)
    axs[1].set_title('RGB image')
    plt.show()
    
    def semD_to_ptcld(depth_image, semantic_image):
        # compute point cloud:
        # Both images has the same resolution
        height, width = depth_image.shape

        # compute indices:
        jj = np.tile(range(width), height)
        ii = np.repeat(range(height), width)

        # Compute constants:
        xx = (jj - CX_DEPTH) / FX_DEPTH
        yy = (ii - CY_DEPTH) / FY_DEPTH

        # transform depth image to vector of z:
        length = height * width
        z = depth_image.reshape(length)

        # compute point cloud
        pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
        cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd) - np.linalg.inv(R).dot(T)
        xx_rgb = ((cam_RGB[:, 0] * FX_RGB) / cam_RGB[:, 2] + CX_RGB + width / 2).astype(int).clip(0, width - 1)
        yy_rgb = ((cam_RGB[:, 1] * FY_RGB) / cam_RGB[:, 2] + CY_RGB).astype(int).clip(0, height - 1)
        #colors = rgb_image[yy_rgb, xx_rgb]
        labels = semantic_image[yy_rgb, xx_rgb].squeeze()
        return pcd, labels
    points, labels = semD_to_ptcld(depth_image, semantic_image)
    #Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])    

    Plot.draw_pc_sem_ins(points, labels)  # visualize ground-truth
    # Convert to Open3D.PointCLoud:
    #pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    #pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    #pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(colors / 255))
    ## Visualize:
    #o3d.visualization.draw_geometries([pcd_o3d])
