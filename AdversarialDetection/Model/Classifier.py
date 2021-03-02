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
class Classifier( nn.Module ):
   def __init__( self, encoder, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.encoder    = encoder
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0

      # Linear Layers
      self.ll1 = nn.Linear( 48 * 4 * 4, 120 )
      self.ll2 = nn.Linear( 120, 10 )

      # Define Loss Criteria
      self.loss = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      encoded = self.encoder.Encode( x )
      encoded = encoded.reshape( -1, 48 * 4 * 4 )
      out = self.ll1( encoded )
      out = self.ll2( out )
      out = torch.sigmoid( out )
      return( out )

   def Train( self, loader, epochs, batch_size ):
      self.train( True ) # Place the model into training mode
      self.encoder.eval( )

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

            # Backward propagation
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
      self.encoder.eval( )
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
            loss = self.loss( outputs, labels )

            _, predicted = outputs.max( 1 )
            testLoss += loss.item( )            
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )
   

