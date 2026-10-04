# Human motion predictor (nioc + ROS 2 Jazzy). Only the dependencies are installed in the image: the source code of
# nioc-neurips and human_kinematic_model is mounted at run time (compose.yaml) and its ROS 2 packages are built by
# the entrypoint.
ARG BASE_IMAGE=ros:jazzy-ros-base
FROM ${BASE_IMAGE}

ARG ZED_INTERFACES_REF=master
# jax (CPU) | "jax[cuda12]" (GPU image, compose service predictor-gpu: needs the NVIDIA container toolkit)
ARG JAX_PACKAGE=jax
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-pip python3-venv python3-colcon-common-extensions python3-yaml \
        ros-${ROS_DISTRO}-visualization-msgs ros-${ROS_DISTRO}-geometry-msgs ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-rmw-fastrtps-cpp ros-${ROS_DISTRO}-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies of the node in a venv that also sees the ROS python packages (rclpy, messages). numpy < 2: the
# ROS 2 Jazzy python extensions of Ubuntu 24.04 are built against numpy 1.26 and do not load with numpy 2, which caps
# JAX at 0.7.1 there (used by the node for the IK only, on the CPU).
RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "numpy<2" scipy pandas matplotlib plotly jax jaxopt
ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv

# Prediction engine (human_motion_predictor/engine.py) in a worker process with its own venv, without ROS: numpy 2 and
# the current JAX, ~3x faster than JAX 0.7.1 on the GPU for this problem (8 goal hypotheses: ~45 vs ~130 ms)
RUN python3 -m venv /opt/jax \
    && /opt/jax/bin/pip install --no-cache-dir numpy scipy pandas matplotlib tqdm "${JAX_PACKAGE}" jaxopt
ENV HMP_WORKER_PYTHON=/opt/jax/bin/python

# ZED message definitions (zed_msgs) for the body-tracking topic of zed-ros2-wrapper
RUN mkdir -p /opt/zed_ws/src \
    && git clone --depth 1 --branch ${ZED_INTERFACES_REF} https://github.com/stereolabs/zed-ros2-interfaces.git \
        /opt/zed_ws/src/zed-ros2-interfaces \
    && apt-get update && rosdep update --rosdistro ${ROS_DISTRO} \
    && rosdep install --from-paths /opt/zed_ws/src -y -i --rosdistro ${ROS_DISTRO} \
    && rm -rf /var/lib/apt/lists/* \
    && . /opt/ros/${ROS_DISTRO}/setup.sh && cd /opt/zed_ws && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /workspace/prophet-ioc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "human_motion_predictor", "predictor.launch.py"]
