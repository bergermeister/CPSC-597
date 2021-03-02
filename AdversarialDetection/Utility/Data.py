import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Data( ):
   def __init__( self, dataset, path, batch_size ):
      self.channels = 0

      if( dataset == 'mnist' ):
         self.channels = 1
         trans = transforms.Compose( [ transforms.Resize( 32 ), # transforms.Scale( 32 ), 
                                       transforms.ToTensor( ), 
                                       transforms.Normalize( ( 0.5 ), ( 0.5 ) ), ] )
         train_dataset = MNIST( root = path, train = True,  transform = trans, download = True )        
         test_dataset  = MNIST( root = path, train = False, transform = trans, download = True )
      elif( dataset == 'cifar10' ):
         self.channels = 3
         trans = transforms.Compose( [ transforms.Resize( 32 ),
                                       transforms.ToTensor( ),
                                       transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) ) ] )
         train_dataset = CIFAR10( root = path, train = True,  transform = trans, download = True )
         test_dataset  = CIFAR10( root = path, train = False, transform = trans, download = True )

      self.train_loader = DataLoader( train_dataset, batch_size, shuffle = True )
      self.test_loader = DataLoader( test_dataset, batch_size, shuffle = True )
