"""TurboDiffusion research layer.

Face identity-preservation / restoration experiments built on top of the
official ``turbodiffusion`` library. This package stages the 4-step pipeline:

  degradation  ->  restoration  ->  evaluation  ->  analysis

Shared helpers live in :mod:`experiments.faceid`, the degradation operations
library in :mod:`experiments.degradation`, and all runnable pipeline scripts
under :mod:`experiments.pipeline`. Paths and constants are centralized in
:mod:`experiments.config`.
"""

__version__ = "1.0.0"
