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

      # Convolutional Layers
      self.cl1 = nn.Sequential( 
         nn.Conv2d( channels, 256, 4, 2, 1 ), # Convolutional Layer 1
         nn.BatchNorm2d( 256 ) )              # Batch Normalization 1
      self.cl2 = nn.Sequential(
         nn.Conv2d( 256, 512, 4, 2, 1 ),      # Convolutional Layer 2
         nn.BatchNorm2d( 512 ) )              # Batch Normalization 2
      self.cl3 = nn.Sequential(
         nn.Conv2d( 512, 1024, 4, 2, 1 ),     # Convolutional Layer 3
         nn.BatchNorm2d( 1024 ) )             # Batch Normalization 3
      
      # Linear Layers
      self.ll1 = nn.Linear( 1024 * 4 * 4, 10 )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      # Debug Commands
      #print( torch.cuda.memory_summary( device = None, abbreviated = False ) )
      #torch.cuda.empty_cache( )
      x = F.relu( self.cl1( x ) )                           # Convolutional Layer 1
      x = F.relu( self.cl2( x ) )                           # Convolutional Layer 2
      x = F.relu( self.cl3( x ) )                           # Convolutional Layer 3
      #x = F.max_pool2d( x, kernel_size = 2, stride = 2 )    # Max Pooling
      x = x.reshape( -1, 1024 * 4 * 4 )                     # Flatten
      x = torch.sigmoid( self.ll1( x ) )                    # Linear Layer 1
      return( x )

   def Train( self, loader, epochs, batch_size ):
      beginTrain = time( )
      totalLoss = 0
      totalCorrect = 0

      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )

      print( "Begin Training..." )
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

         print( f'epoch #{epoch} | loss: {loss}' )
      print( f"End Training... loss: {totalLoss}" )

   def GetVariable( self, arg ):
      var = None
      if( self.cudaEnable == True ):
         var = Variable( arg ).cuda( 0 )
      else:
         var = Variable( arg )
      return( var )