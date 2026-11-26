"""The specification: a model as pure data. Blocks, ports, parameters, shocks, equation trees, data links, restrictions, edits and the difference between two versions.

No JAX in this layer, so a specification can be read, edited and compared without any numerical machinery.
"""

from mutatis.spec.checks import Finding, check
from mutatis.spec.diff import diff
from mutatis.spec.edits import EDITS, Edit, EditError, apply_edits, edit_from_dict
from mutatis.spec.model import Block, DataLink, Model, Parameter, Restriction, Shock
from mutatis.spec.numbers import NumberSite, find_numbers
from mutatis.spec.yaml_io import SpecError, dump_model, load_model
