from setuptools import setup, find_packages

setup(
    name="lidar_det",
    version="2.1",
    author="Jinzheng Guang",
    author_email="guangjinzheng@mail.nankai.edu.cn",
    packages=find_packages(
        include=["lidar_det", "lidar_det.*", "lidar_det.*.*"]
    ),
    license="LICENSE.txt",
    description="RPEA: A Residual Path Network with Efficient Attention for 3D Pedestrian Detection from LiDAR Point Clouds.",
)
