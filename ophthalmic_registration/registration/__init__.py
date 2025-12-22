"""Registration pipeline for ophthalmic images."""

from ophthalmic_registration.registration.sift_aligner import SiftAligner
from ophthalmic_registration.registration.ecc_aligner import EccAligner
from ophthalmic_registration.registration.registration_pipeline import (
    RegistrationPipeline,
    CoarseAlignmentMethod,
)
from ophthalmic_registration.registration.feature_aligner import (
    FeatureAligner,
    FeatureDetector,
)

# Optional Elastix support
try:
    from ophthalmic_registration.registration.elastix_aligner import (
        ElastixAligner,
        ElastixTransformType,
        ElastixMetric,
        is_simpleitk_available,
        is_elastix_available,
    )
    _ELASTIX_EXPORTS = [
        "ElastixAligner",
        "ElastixTransformType",
        "ElastixMetric",
        "is_simpleitk_available",
        "is_elastix_available",
    ]
except ImportError:
    _ELASTIX_EXPORTS = []

__all__ = [
    "SiftAligner",
    "EccAligner",
    "RegistrationPipeline",
    "CoarseAlignmentMethod",
    "FeatureAligner",
    "FeatureDetector",
] + _ELASTIX_EXPORTS
