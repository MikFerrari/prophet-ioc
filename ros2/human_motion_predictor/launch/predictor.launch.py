"""Human motion predictor: python launch with the default parameters of config/predictor.yaml, plus the goal
locations of the cell (goals_file).

    ros2 launch human_motion_predictor predictor.launch.py goals_file:=<goals.yaml>     # ZED skeletons (ik.py)
    ros2 launch human_motion_predictor predictor.launch.py input:=zed replay:=true      # CARI session replay of the raw
    ros2 launch human_motion_predictor predictor.launch.py input:=joints replay:=true   # ZED keypoints / IK angles
        (session and goals exported on the host by evaluation/export_replay.py)
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def nodes(context):
    arg = lambda name: context.launch_configurations[name]
    replay = arg("replay").lower() == "true"
    goals = arg("goals_file") or (arg("trial_file").replace(".npz", "_goals.yaml") if replay else "")
    params = [arg("config")] + ([goals] if goals else []) + [{"input": arg("input"),
                                                              "params_file": arg("params_file")}]
    out = [Node(package="human_motion_predictor", executable="predictor", name="human_motion_predictor",
                output="screen", parameters=params)]
    if replay:
        out.append(Node(package="human_motion_predictor", executable="replay_cari", name="cari_replay",
                        output="screen", parameters=[{"trial_file": arg("trial_file"), "mode": arg("input")}]))
    return out


def generate_launch_description():
    config = str(Path(get_package_share_directory("human_motion_predictor")) / "config" / "predictor.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config, description="parameter file of the predictor"),
        DeclareLaunchArgument("input", default_value="zed", description="zed | joints"),
        DeclareLaunchArgument("params_file", default_value="", description="train.py params.json (optional)"),
        DeclareLaunchArgument("goals_file", default_value="",
                              description="goal locations of the cell (parameter file, e.g. <session>_goals.yaml); "
                                          "default: those of the replayed session with replay:=true, else none"),
        DeclareLaunchArgument("replay", default_value="false", description="also start the CARI session replay"),
        DeclareLaunchArgument("trial_file", default_value="/workspace/prophet-ioc/output/replay/sub_4_FAST.npz",
                              description="session of the replay (evaluation/export_replay.py)"),
        OpaqueFunction(function=nodes),
    ])
