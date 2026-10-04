# Predicted motion of the 9 upper-body joints of a person (human_motion_predictor), for one goal hypothesis
std_msgs/Header header             # stamp: time of the last observed frame; frame_id: frame of the input keypoints
int32 person_id                    # tracking id of the person (-1: unknown)
string reaching_hand               # "right" or "left"
string[] joint_names               # head, chest, pelvis, left_shoulder, left_elbow, left_wrist, right_shoulder, ...
float64[] time_from_start          # times in s after header.stamp, 0 to prediction_time every prediction_dt
geometry_msgs/Point[] positions    # len(time_from_start) x len(joint_names) positions in m, step-major
float64[] wrist_covariance         # len(time_from_start) x 9: 3x3 position covariance of the reaching wrist (row-major, m^2)
float64[] elbow_covariance         # len(time_from_start) x 9: the same for the reaching elbow
float64[] passive_wrist_covariance # len(time_from_start) x 9: the same for the other wrist
string goal_name                   # hypothesis of this prediction: a goal of the cell, or "idle" (coming to rest)
geometry_msgs/Point goal           # its final goal (idle: the rest position)
geometry_msgs/Point target         # target of the optimal-control problem: the goal, or (target_is_temporary) the point
bool target_is_temporary           #   of the minimum-jerk path to the goal reached at the end of the prediction window
float64 arrival_time               # expected arrival at the goal, s after header.stamp (may exceed the prediction window;
                                   #   the posture is held after min(arrival_time, prediction window))
float64 probability                # posterior probability of the hypothesis
string[] hypothesis_names          # posterior over all the hypotheses (published with the most probable one)
float64[] hypothesis_probabilities
float64 computation_time           # s spent on the prediction of all hypotheses and the goal inference
