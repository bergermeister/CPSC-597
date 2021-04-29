import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time
from Utility.ProgressBar import ProgressBar

class MetaClassifier( nn.Module ):
   def __init__( self, cudaEnable ):
      super( ).__init__( )