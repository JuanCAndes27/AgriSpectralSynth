"""
Crown-delineation agent: chooses, per scene, which method to use and
learns from a reward computed against annotated crowns.

    groundtruth  annotations (VOC XML, CSV boxes, WKT polygons)
    evaluation   IoU matching, F1, count error, reward
    actions      the delineation methods the agent can choose ("arms")
    features     scene descriptors (the context)
    rewards      offline reward table (every arm on every scene)
    bandit       contextual bandit policies and the exported agent
    experiment   site-grouped cross-validation protocol
"""

from .actions import ARMS, Arm, run_arm
from .bandit import LinUCB, TrainedAgent
from .evaluation import RewardWeights, reward, score_detections
from .groundtruth import GroundTruth, load_annotations

__all__ = ["ARMS", "Arm", "GroundTruth", "LinUCB", "RewardWeights", "TrainedAgent",
           "load_annotations", "reward", "run_arm", "score_detections"]
