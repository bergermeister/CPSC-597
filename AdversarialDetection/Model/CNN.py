##
# @package CNN
#
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from time import time
from Utility.ProgressBar import ProgressBar

## Deep Convolutional Neural Network
# 
# @details
# @par
# This class provides the Deep Convolutional Neural Network with 3 Convolutional Layers and 1 Linear Layer.
class CNN( nn.Module ):
   def __init__( self, channels, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0
      self.interm     = []

      # Convolutional Layers
      self.cl1 = nn.Sequential( 
         nn.Conv2d( channels, 16, 3, 1, 0 ), # Convolutional Layer 1 (32 x 32) -> (30 x 30)
         nn.BatchNorm2d( 16 ) )              # Batch Normalization 1 
      self.cl2 = nn.Sequential(
         nn.Conv2d( 16, 32, 3, 1, 0 ),       # Convolutional Layer 2 (30 x 30) -> (28 x 28)
         nn.BatchNorm2d( 32 ) )              # Batch Normalization 2
      self.cl3 = nn.Sequential(
         nn.Conv2d( 32, 64, 3, 1, 0 ),       # Convolutional Layer 3 (28 x 28) -> (26 x 26)
         nn.BatchNorm2d( 64 ) )              # Batch Normalization 3            

      # Pooling
      self.mp = nn.MaxPool2d( 2 )            # Max Pool Layer (26 x 26) -> (13 x 13)

      # Linear Layers
      self.ll1 = nn.Linear( 64 * 13 * 13, 10 )

      # Define Loss Criteria
      self.loss = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      self.interm = []
      dl1 = nn.ConvTranspose2d(  16, 1, 1, 1, 0 ).cuda( )
      dl1.weight.data.fill_( 1 )
      dl1.bias.data.fill_( 0 )
      dl2 = nn.ConvTranspose2d(  32, 16, 1, 1, 0 ).cuda( )
      dl2.weight.data.fill_( 1 )
      dl2.bias.data.fill_( 0 )
      dl3 = nn.ConvTranspose2d( 64, 32, 1, 1, 0 ).cuda( )
      dl3.weight.data.fill_( 1 )
      dl3.bias.data.fill_( 0 )

      x = F.relu( self.cl1( x ) )                           # Convolutional Layer 1
      self.interm.append( dl1( x ).detach( ) )
      x = F.relu( self.cl2( x ) )                           # Convolutional Layer 2
      self.interm.append( dl1( dl2( x ) ).detach( ) )
      x = F.relu( self.cl3( x ) )                           # Convolutional Layer 3
      self.interm.append( dl1( dl2( dl3( x ) ) ).detach( ) )
      x = self.mp( x )                                      # Max pool
      x = x.reshape( -1,  64 * 13 * 13 )                    # Flatten
      x = torch.sigmoid( self.ll1( x ) )                    # Linear Layer 1

      return( x )

   def Train( self, loader, epochs, batch_size ):
      self.train( True ) # Place the model into training mode

      progress = ProgressBar( 40, 80 )
      beginTrain = time( )
      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )

      print( "Begin Training..." )
      for epoch in range( epochs ):
         beginEpoch = time( )

         trainLoss = 0
         total     = 0
         correct   = 0
         for batchIndex, ( images, labels ) in enumerate( loader ):
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            outputs = self( images )
            loss = self.loss( outputs, labels )
            #loss = F.cross_entropy( output, labels )

            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights

            _, predicted = outputs.max( 1 )
            trainLoss    += loss.item( )
            total        += labels.size( 0 )
            correct      += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( self.epochs + epoch, trainLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Training...' )

   def Test( self, loader, batch_size ):
      self.eval( ) # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )

      testLoss = 0
      total    = 0
      correct  = 0

      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( images, labels ) in enumerate( loader ):
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            outputs = self( images )
            loss    = self.loss( outputs, labels )

            _, predicted = outputs.max( 1 )
            testLoss += loss.item( )            
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )
   
   def Evaluate( self ):
      return( 0 )

   def GetVariable( self, arg ):
      var = None
      if( self.cudaEnable == True ):
         var = Variable( arg ).cuda( )
      else:
         var = Variable( arg )
      return( var )
