import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from time import time

# Deep Convolutional Neural Network
class CNN( nn.Module ):
   def __init__( self, channels, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.cudaEnable = bool( cudaEnable )

      # Convolutional Layer 1
      self.cl1 = nn.Sequential( 
         nn.ConvTranspose2d( channels, 256, 4, 2, 1 ), # Convolutional Layer 1
         nn.BatchNorm2d( 256 ) )                       # Batch Normalization 1

      # Convolutional Layer 2
      self.cl2 = nn.Sequential(
         nn.ConvTranspose2d( 256, 512, 4, 2, 1 ),      # Convolutional Layer 2
         nn.BatchNorm2d( 512 ) )                       # Batch Normalization 2

      # Convolutional Layer 3
      self.cl3 = nn.Sequential(
         nn.ConvTranspose2d( 512, 1024, 4, 2, 1 ),     # Convolutional Layer 3
         nn.BatchNorm2d( 1024 ) )                      # Batch Normalization 3
      
      # Linear Layers
      self.ll1 = nn.Linear( 1024 * 4 * 4, 120 )
      self.ll2 = nn.Linear( 120, 10 )
            
   def forward( self, x ):
      # Run through Convolutional Layer 1
      #torch.cuda.empty_cache( )
      self.cl1.cuda( )
      x = F.relu( self.cl1( x ) )
      self.cl1.cpu( )
      
      # Run through Convolutional Layer 2
      #print( torch.cuda.memory_summary( device = None, abbreviated = False ) )
      #torch.cuda.empty_cache( )
      self.cl2.cuda( )
      #print( torch.cuda.memory_summary( device = None, abbreviated = False ) )
      x = F.relu( self.cl2( x ) )
      self.cl2.cpu( )

      # Run through Convolutional Layer 3
      #print( torch.cuda.memory_summary( device = None, abbreviated = False ) )
      #torch.cuda.empty_cache( )
      self.cl3.cuda( )
      #print( torch.cuda.memory_summary( device = None, abbreviated = False ) )
      x = F.relu( self.cl3( x ) )
      self.cl3.cpu( )

      # Max Pooling
      x = F.max_pool2d( x, kernel_size = 2, stride = 2 )
      
      # Flatten
      x = x.reshape( -1, 1024 )

      # Run through Linear Layer 1
      x = self.ll1( x )
      x = F.relu( x )

      # Run through Linear Layer 2
      x = self.ll2( x )
      x = F.sigmoid( x )

      # Run through Linear Layer 3
      #x = self.ll3( x )
      #x = F.sigmoid( x )

      return( x )

   def Train( self, loader, epochs, batch_size ):
      beginTrain = time( )
      totalLoss = 0
      totalCorrect = 0

      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )

      for epoch in range( epochs ):
         beginEpoch = time( )
         
         for ( images, labels ) in loader:
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            preds = self( images )
            loss = F.cross_entropy( preds, labels )

            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights

            totalCorrect += preds.argmax( dim = 1 ).eq( labels ).sum( ).item( )
            totalLoss += loss.item( )

         print( f'epoch #{epoch} | loss: {totalLoss}' )

   def GetVariable( self, arg ):
      var = None
      if( self.cudaEnable == True ):
         var = Variable( arg ).cuda( 0 )
      else:
         var = Variable( arg )
      return( var )