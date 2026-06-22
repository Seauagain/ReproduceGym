"""ReproduceGym verifier: score a finished workspace by RECOMPUTING claim metrics
from the agent's raw artifacts. The verifier never reads any agent-declared
verdict, score, or reward -- the verdict is derived here from recomputed numbers.
"""

from reproducegym.verifier.engine import VerifierError, recompute, safe_eval

__all__ = ["VerifierError", "recompute", "safe_eval"]
