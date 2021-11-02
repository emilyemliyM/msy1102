# transformation between Cartesian coordinates and polar coordinates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def cart2polar(input_xyz): # (91943,3)
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2) # （91943，）
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    ffff=np.stack((rho,phi,input_xyz[:,2]),axis=1)
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)


  # Skip every n points
def get_label(file_path):
    # label_file_path = os.path.join("xt", scene_name, 'gt', sample_id + '.txt')
    return np.loadtxt(file_path)

def plot_pt(points ,label ):

    fig = plt.figure()
    skip = 2
    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip) # skip points to prevent crash
    x = points[point_range, 0]
    z = points[point_range, 1]
    y = points[point_range, 2]
    map_color = {0:'r', 1:'g', 2:'b'}
    Label= label[point_range,-1]
    # Label =np.loadtxt("E:\polar1101\data\\nuscenes_moving_obstacle_detection\scene-0001\gt\000000.txt")

    Color = list(map(lambda x: map_color[x], Label))
    # print( Color.shape)
    # map_maker = {-1:'o', 1:'v'}
    # makers = list(map(lambda x:map_maker[x], Label))
    #  c: height data for color
    ax.scatter(x,   y,   z,   c=Color, marker="o")
    # ax.axis('scaled')  # {equal, scaled}
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('part_seg')
    ax.axis('off')          # 设置坐标轴不可见
    ax.grid(False)          # 设置背景网格不可见
    plt.show()

if __name__ == '__main__':
    points = np.fromfile('E://polar1101//data//nuscenes_moving_obstacle_detection//scene-0001//lidar//000000.bin').reshape(-1, 3)
    # print( points.shape)
    label =get_label("E:/polar1101/data/nuscenes_moving_obstacle_detection/scene-0001/gt/000000.txt")
    plot_pt(points, label)