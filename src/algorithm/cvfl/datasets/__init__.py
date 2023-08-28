from .covtype import CovType
from .epsilon import Epsilon
from .gisette import Gisette
from .higgs import Higgs
from .letter import Letter
from .msd import MSD
from .radar import Radar
from .realsim import Realsim
from .cifar10 import CIFAR10
from .mnist import MNIST

from os.path import dirname, abspath
import sys

dir_path = dirname(abspath(__file__))
sys.path.append(dir_path)
sys.path.append(dir_path + "/vertibench")
