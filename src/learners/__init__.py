from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .nq_learner_data_augmentation import NQLearnerDataAugmentation
from .saleq_wm_learner import SaleqWMLearner  # ported from marl_project3

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner_data_augmentation"] = NQLearnerDataAugmentation
REGISTRY["saleq_wm_learner"] = SaleqWMLearner

from .ablation_mixstate_learner import AblationMixStateLearner  # ABLATION arms 5a/5b: mixer global-state input
REGISTRY["ablation_mixstate_learner"] = AblationMixStateLearner

from .design_wm_learner import DesignWMLearner  # DESIGN-SPACE study (E3): configurable mixer state input
REGISTRY["design_wm_learner"] = DesignWMLearner
