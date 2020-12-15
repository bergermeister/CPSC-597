import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Data( ):
   def __init__( self, dataset, path, batch_size ):
      if( dataset == 'mnist' ):
         trans = transforms.Compose( [ transforms.Resize( 32 ), # transforms.Scale( 32 ), 
                                       transforms.ToTensor( ), 
                                       transforms.Normalize( ( 0.5 ), ( 0.5 ) ), ] )
         train_dataset = MNIST( root = path, train = True,  transform = trans, download = True )        
         test_dataset  = MNIST( root = path, train = False, transform = trans, download = True )

      self.train_loader = DataLoader( train_dataset, batch_size, shuffle = True )
      self.test_loader = DataLoader( test_dataset, batch_size, shuffle = True )
