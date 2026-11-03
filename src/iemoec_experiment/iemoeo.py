"""IE-MOEO construction entry point; preserve the historical founder engine."""
from .iemoeo_founder import (
    IEMOEORunner as FounderRunner, ReferenceDirectionSurvival,
    clip_to_bounds, merge_populations, variable_blocks,
)


def IEMOEORunner(problem, case, initial_X=None, on_checkpoint=None,
                 on_outer_selection=None, on_evaluation=None):
    """Keep existing constructor calls while choosing the registered architecture."""
    runner_class = FounderRunner
    if case.iemoeo.architecture == "s3_attribution":
        from .iemoeo_attribution import AttributionRunner
        runner_class = AttributionRunner
    return runner_class(problem, case, initial_X, on_checkpoint,
                        on_outer_selection, on_evaluation)
