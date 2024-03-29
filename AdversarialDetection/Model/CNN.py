##
# @package CNN
#
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import utils
from torchvision import transforms
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
      self.indices    = {}

      # Convolutional Layers
      self.cl1 = nn.Conv2d( self.channels, 16, 5, 1, 1 ) # Convolutional Layer 1 (16, 16, 16)
      self.cl2 = nn.Conv2d(            16, 32, 5, 1, 1 ) # Convolutional Layer 2 (32,  8 , 8)
      self.cl3 = nn.Conv2d(            32, 64, 5, 1, 1 ) # Convolutional Layer 3 (64,  4,  4)
      self.mp  = nn.MaxPool2d( 2, stride = 2, return_indices = True ) # Max Pooling
      self.ll1 = nn.Linear( 64 * 2 * 2, 10 )
      self.act = nn.Softmax( dim = 1 )

      # Define Loss Criteria
      self.lossFunc = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      out, self.indices[ 'mp1' ] = self.mp( F.relu( self.cl1(   x ) ) )
      out, self.indices[ 'mp2' ] = self.mp( F.relu( self.cl2( out ) ) )
      out, self.indices[ 'mp3' ] = self.mp( F.relu( self.cl3( out ) ) )
      out = out.reshape( -1, 64 * 2 * 2 )
      out = self.ll1( out )
      out = self.act( out )
      return( out )

   def Load( self, path ):
      state = torch.load( path )
      self.load_state_dict( state[ 'model' ] )
      self.accuracy = state[ 'acc' ]
      self.epochs   = state[ 'epoch' ]

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
            loss = self.lossFunc( outputs, labels )

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
            loss    = self.lossFunc( outputs, labels )

            _, predicted = outputs.max( 1 )
            testLoss += loss.item( )            
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )
   
   def TestGaussian( self, loader, batch_size ):
      self.eval( ) # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )

      #gaussian = transforms.GaussianBlur( 5, sigma=( 0.1, 2.0 ) )
      gaussian = transforms.GaussianBlur( 5, sigma=( 2.0 ) )

      testLoss = 0
      total    = 0
      correct  = 0

      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( images, labels ) in enumerate( loader ):
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            blurred = gaussian( images )
            outputs = self( blurred )
            loss    = self.lossFunc( outputs, labels )

            _, predicted = outputs.max( 1 )
            testLoss += loss.item( )            
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )

   def TestGaussianHighpass( self, loader, batch_size ):
      self.eval( ) # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )

      #gaussian = transforms.GaussianBlur( 5, sigma=( 0.1, 2.0 ) )
      gaussian = transforms.GaussianBlur( 5, sigma=( 2.0 ) )
      
      testLoss = 0
      total    = 0
      correct  = 0

      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( images, labels ) in enumerate( loader ):
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            blurred = gaussian( images )
            highpass = images - blurred
            outputs = self( highpass )
            loss    = self.lossFunc( outputs, labels )

            _, predicted = outputs.max( 1 )
            testLoss += loss.item( )            
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )