from models.zoo.icam.policy import ICAMPolicy
from models.zoo.dgnn.policy import DGNNPolicy
POLICY_REGISTRY = {
    "icam": ICAMPolicy,
    "dgnn": DGNNPolicy,
}

def get_policy(name: str, **kwargs):
    policy_cls = POLICY_REGISTRY.get(name, None)
    if policy_cls is None:
        raise ValueError(
            f"policy {name} not supported yet"
        )
    return policy_cls(**kwargs)