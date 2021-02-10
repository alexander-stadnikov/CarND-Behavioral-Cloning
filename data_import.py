import os
import csv
from dataclasses import dataclass
from typing import List

from frame import Frame


@dataclass
class SimulatorDialect(csv.Dialect):
    """ Describes the dialect of input data. """
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_NONE
    delimiter = ','
    lineterminator = '\n'


def read_csv(path: str) -> List[Frame]:
    """
    Imports frames from the provided CSV file.
    """
    out = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, SimulatorDialect())
        for line in reader:
            out.append(Frame(line))
    return out
