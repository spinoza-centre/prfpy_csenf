import scipy
from packaging import version
MIN_SCIPY_VERSION = '1.8.0'

# Check the scipy version
if version.parse(scipy.__version__) < version.parse(MIN_SCIPY_VERSION):
    raise ImportError(f"Your installed scipy version ({scipy.__version__}) is too old. "
                      f"Please upgrade to scipy {MIN_SCIPY_VERSION} or higher.")