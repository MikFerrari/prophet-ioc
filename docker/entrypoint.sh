#!/bin/bash
# Sources ROS 2 and the ZED messages, puts the mounted source on the python path and builds its ROS 2 packages
# (ros2/) into /ros2_ws (a volume, so the build artifacts stay out of the source). SKIP_BUILD=1 skips the build.
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
source /opt/zed_ws/install/setup.bash

export PROPHET_ROOT=${PROPHET_ROOT:-/workspace/prophet-ioc}
export HKM_ROOT=${HKM_ROOT:-/workspace/human_kinematic_model}
# the ROS 2 executables run the system python3 (colcon shebang): give it the venv packages (jax, ...) too
VENV_SITE=$(/opt/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=${PROPHET_ROOT}:${HKM_ROOT}/scripts:${VENV_SITE}:${PYTHONPATH}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    colcon --log-base /ros2_ws/log build --base-paths "${PROPHET_ROOT}/ros2" \
        --build-base /ros2_ws/build --install-base /ros2_ws/install --symlink-install \
        --event-handlers console_cohesion- status- > /ros2_ws/colcon_build.log 2>&1 \
        || { cat /ros2_ws/colcon_build.log; exit 1; }
fi
source /ros2_ws/install/setup.bash
exec "$@"
