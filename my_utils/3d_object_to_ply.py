import numpy as np
import os

# 创建LS文件夹
ls_folder = 'G:/database/LS_PCQA/LS'
os.makedirs(ls_folder, exist_ok=True)

# 待处理的3D对象文件目录
objects_dir = 'G:/database/LS_PCQA/cloudpoint'
print(os.listdir(objects_dir))
# 遍历每个3D对象文件
for filename in os.listdir(objects_dir):
    if filename.endswith('.obj'):
        # 读取3D对象文件
        filepath = os.path.join(objects_dir, filename)
        # 执行您的操作（转换为点云数据）
        vertices = read_object_file(filepath)
        print("Number of vertices:", len(vertices))
        print("\n")

        # 创建.ply文件并写入文件头
        ply_file_path = os.path.join(ls_folder, filename[:-4] + '.ply')
        ply_file = open(ply_file_path, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % len(vertices))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # 将点云数据写入.ply文件
        for vertex in vertices:
            ply_file.write("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))

        # 关闭.ply文件
        ply_file.close()


