from .release import __version__
from .install import install
import os
import julia
julia.Julia(compiled_modules=False)
