from glob import glob

from setuptools import find_packages, setup

package_name = "human_motion_predictor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Michele Ferrari",
    maintainer_email="michele.ferrari@unibs.it",
    description="Human upper-body motion prediction with the 19-DOF kinematic model of prophet_ioc.",
    license="GPL-3.0-only",
    entry_points={
        "console_scripts": [
            "predictor = human_motion_predictor.node:main",
            "replay_cari = human_motion_predictor.replay_cari:main",
        ],
    },
)
