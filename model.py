"""Public model factory for the stDiff and IMD-T pipelines."""

if __package__:
    from .gene_attention.model import *
    from .gene_attention.model import __all__
else:
    from gene_attention.model import *
    from gene_attention.model import __all__
