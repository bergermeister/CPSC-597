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
   def __init__( self, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0
      self.activation = {}

      # Linear Layers
      self.ll1 = nn.Linear( 16 * 15 * 15, 10 )
      self.ll2 = nn.Linear( 32 *  6 *  6, 10 )
      self.ll3 = nn.Linear( 64 *  2 *  2, 10 )
      self.mp  = nn.MaxPool2d( 2, stride = 2 ) # Max Pooling
      self.act = nn.Softmax( dim = 1 )

      # Define Loss Criteria
      self.lossFunc = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      cnn   = x[ 'model' ]
      input = x[ 'input' ]

      # Register Hooks
      cnn.cl1.register_forward_hook( self.Hook( 'cl1' ) )
      cnn.cl2.register_forward_hook( self.Hook( 'cl2' ) )
      cnn.cl3.register_forward_hook( self.Hook( 'cl3' ) )

      # Forward Pass through CNN
      with torch.no_grad():
         cnn( input )

      # Classify each convolutional layer
      out1 = self.act( self.ll1( self.mp( F.relu( self.activation[ 'cl1' ] ) ).reshape( -1, 16 * 15 * 15 ) ) )
      out2 = self.act( self.ll2( self.mp( F.relu( self.activation[ 'cl2' ] ) ).reshape( -1, 32 *  6 *  6 ) ) )
      out3 = self.act( self.ll3( self.mp( F.relu( self.activation[ 'cl3' ] ) ).reshape( -1, 64 *  2 *  2 ) ) )

      return( out1, out2, out3 )

   def Load( self, path ):
      state = torch.load( path )
      self.load_state_dict( state[ 'model' ] )
      self.accuracy = state[ 'acc' ]
      self.epochs   = state[ 'epoch' ]

   def Train( self, cnn, loader, epochs, batch_size ):
      self.train( True )   # Place the model into training mode
      cnn.eval( )          # Place CNN into evaluation mode

      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )
      progress = ProgressBar( 40, 80 )
      beginTrain = time( )
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
            out1, out2, out3 = self( { 'model' : cnn, 'input' : images } )
            loss1 = self.lossFunc( out1, labels )
            loss2 = self.lossFunc( out2, labels )
            loss3 = self.lossFunc( out3, labels )
            
            # Backward propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss1.backward( )       # Calculate Gradients
            loss2.backward( )       # Calculate Gradients
            loss3.backward( )       # Calculate Gradients
            optimizer.step( )       # Update Weights
            
            # Logging
            _, predicted1 = out1.max( 1 )
            _, predicted2 = out2.max( 1 )
            _, predicted3 = out3.max( 1 )
            trainLoss    += loss1.item( ) + loss2.item( ) + loss3.item( )
            total        += ( labels.size( 0 ) * 3 )
            correct      += predicted1.eq( labels ).sum( ).item( ) + predicted2.eq( labels ).sum( ).item( ) + predicted3.eq( labels ).sum( ).item( )
            progress.Update( batchIndex, len( loader ), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( self.epochs + epoch, trainLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Training...' )

   def Test( self, cnn, loader, batch_size ):
      self.eval( ) # Place the model into test/evaluation mode
      cnn.eval( )          # Place CNN into evaluation mode
      progress = ProgressBar( 40, 80 )

      testLoss = 0
      total    = 0
      correct  = 0
      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( images, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            out1, out2, out3 = self( { 'model' : cnn, 'input' : images } )
            loss1 = self.lossFunc( out1, labels )
            loss2 = self.lossFunc( out2, labels )
            loss3 = self.lossFunc( out3, labels )

            # Logging
            _, predicted1 = out1.max( 1 )
            _, predicted2 = out2.max( 1 )
            _, predicted3 = out3.max( 1 )
            testLoss    += loss1.item( ) + loss2.item( ) + loss3.item( )
            total        += ( labels.size( 0 ) * 3 )
            correct      += predicted1.eq( labels ).sum( ).item( ) + predicted2.eq( labels ).sum( ).item( ) + predicted3.eq( labels ).sum( ).item( )
            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )
   
   def Hook( self, name ):
      def hook( model, input, output ):
         self.activation[ name ] = output.detach( )
      return( hook )
