from .blob import Blob
from .iris import Iris
from .diabetes import Diabetes
from .bostonhousing import BostonHousing
from .wine import Wine
from .breastcancer import BreastCancer
from .qsar import QSAR
from .mimic import MIMICL, MIMICM
from .mnist import MNIST
from .cifar import CIFAR10
from .modelnet import ModelNet40
from .shapenet import ShapeNet55

from .covtype import CovType
from .epsilon import Epsilon
from .gisette import Gisette
from .higgs import Higgs
from .letter import Letter
from .msd import MSD
from .radar import Radar
from .realsim import Realsim
from .cifar10_vb import CIFAR10_VB
from .mnist_vb import MNIST_VB
from .vehicle import Vehicle
from .wide import Wide

from os.path import dirname, abspath, join
import sys

dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(join(dir_path, 'vertibench'))

from .utils import *
__all__ = ('Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MIMICL', 'MIMICM'
           'MNIST', 'CIFAR10', 'ModelNet40', 'ShapeNet55', 'MSD', 'CovType', 'Higgs', 'Gisette', 'Letter', 'Radar', 'Epsilon', 'Realsim', "MNIST_VB", "CIFAR10_VB", "Wide", "Vehicle")