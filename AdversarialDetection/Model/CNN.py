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
      self.channels = channels
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0
      self.interm     = []

      # Convolutional Layers
      self.cl1 = nn.Conv2d( self.channels, 12, 4, 2, 1 ) # Convolutional Layer 1 (12, 16, 16)
      self.bn1 = nn.BatchNorm2d( 12 )                    # Batch Normalization 1 
      self.cl2 = nn.Conv2d( 12, 24, 4, 2, 1 )            # Convolutional Layer 2 (24,  8 , 8)
      self.bn2 = nn.BatchNorm2d( 24 )                    # Batch Normalization 2
      self.cl3 = nn.Conv2d( 24, 48, 4, 2, 1 )            # Convolutional Layer 3 (48,  4,  4)
      self.bn3 = nn.BatchNorm2d( 48 )                    # Batch Normalization 3            
      self.mp  = nn.MaxPool2d( 2, 2 )                    # Max Pool Layer (20 x 20) -> (10 x 10)

      # Linear Layers
      self.ll1 = nn.Linear( 48 * 2 * 2, 10 )

      # Define Loss Criteria
      self.loss = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      out = self.cl1( x )   
      out = self.bn1( out )
      out = F.relu( out )
      out = self.cl2( out )
      out = self.bn2( out )
      out = F.relu( out )
      out = self.cl3( out )
      out = self.bn3( out )
      out = F.relu( out )
      out = self.mp( out )
      out = out.reshape( -1, 48 * 2 * 2 )
      out = self.ll1( out )
      out = torch.sigmoid( out )
      return( out )

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
            # Forward pass
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            outputs = self( images )
            loss = self.loss( outputs, labels )

            # Backward Propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights

            # Logging
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
   
