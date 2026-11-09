# ==============================================================================
# Imports
# ==============================================================================
from __future__ import annotations

import os
import xml.etree.ElementTree as ET


# ==============================================================================
# SBML reading helpers (ElementTree)
# ==============================================================================
# > SBML core, fbc and qual tags and attributes are matched by their local name:
# > "{namespace}name" -> "name", whatever the version of the package.
# ==============================================================================
def local(name: str) -> str:
    return name.rpartition("}")[2]


def attr(element: ET.Element, name: str) -> str | None:
    for key, value in element.attrib.items():
        if local(key) == name:
            return value
    return None


def read_model(sbml_file: str) -> ET.Element:
    # ~ The <model> element of an SBML file
    if not os.path.isfile(sbml_file):
        raise FileNotFoundError(f"SBML file not found: {sbml_file}")
    for element in ET.parse(sbml_file).getroot():
        if local(element.tag) == "model":
            return element
    raise ValueError(f"No model in SBML file: {sbml_file}")


def sections(model: ET.Element) -> dict[str, ET.Element]:
    # ~ The lists of a model (listOfSpecies, listOfReactions...) by local name
    return {local(element.tag): element for element in model}
