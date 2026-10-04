"""Data-driven baselines learned from demonstrations (training reaches): ProMP and DMP.

Both predict the 9 upper-body joints of the CARI v2 evaluation (eval.py, methods "promp" and "dmp") in Cartesian
space, from the observed prefix of a reach, the reaching-wrist goal and the arrival time, like the goal-directed
baselines of prophet_ioc.infer; ProMP also returns position covariances. See promp.py, dmp.py and common.py.
"""

from prophet_ioc.baselines.common import JOINTS, BaselinePrediction, Reach, canonical_frame
from prophet_ioc.baselines.dmp import DMP, DMPBaseline
from prophet_ioc.baselines.promp import ProMP, ProMPBaseline

__all__ = ["JOINTS", "BaselinePrediction", "Reach", "canonical_frame", "DMP", "DMPBaseline", "ProMP",
           "ProMPBaseline", "BASELINES"]

# eval.py method name -> predictor class (each has fit(reaches, **options) and predict(obs, hand, target, fut_times, dt))
BASELINES = {"promp": ProMPBaseline, "dmp": DMPBaseline}
