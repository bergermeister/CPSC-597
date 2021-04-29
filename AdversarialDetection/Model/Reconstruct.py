
##
# @package CNN
#
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import utils
from time import time
from Utility.ProgressBar import ProgressBar

## Deep Convolutional Neural Network
# 
# @details
# @par
# This class provides the Deep Convolutional Neural Network with 3 Convolutional Layers and 1 Linear Layer.
class Reconstruct( nn.Module ):
   def __init__( self, channels, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.channels = channels
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0
      self.activation = {}

      # Reconstruction Networks
      self.dl11 = nn.ConvTranspose2d( 16, self.channels, 5, 1, 1 )
      self.dl21 = nn.ConvTranspose2d( 32,            16, 5, 1, 1 )
      self.dl22 = nn.ConvTranspose2d( 16, self.channels, 5, 1, 1 ) 
      self.dl31 = nn.ConvTranspose2d( 64,            32, 5, 1, 1 )
      self.dl32 = nn.ConvTranspose2d( 32,            16, 4, 1, 0 )
      self.dl33 = nn.ConvTranspose2d( 16, self.channels, 5, 1, 1 )
      self.mp   = nn.MaxUnpool2d( 2, 2 )
      self.relu = nn.ReLU( True )
      self.sigm = nn.Sigmoid( )

      # Define Loss Criteria
      self.lossFunc =  nn.MSELoss( )

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

      # Decode first convolutional layer
      out1 = self.sigm( self.dl11( self.activation[ 'cl1' ] ) )

      # Decode second convolutional layer
      out2 = self.relu( self.dl21( self.activation[ 'cl2' ] ) )
      out2 = self.mp( out2, cnn.indices[ 'mp1' ] )
      out2 = self.sigm( self.dl22( out2 ) )

      # Decode third convolutional layer
      out3 = self.relu( self.dl31( self.activation[ 'cl3' ] ) )
      out3 = self.mp( out3, cnn.indices[ 'mp2' ] )
      out3 = self.relu( self.dl32( out3 ) )
      out3 = self.mp( out3, cnn.indices[ 'mp1' ] )
      out3 = self.sigm( self.dl33( out3 ) )

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

         images = 0
         out1   = 0
         out2   = 0
         out3   = 0
         total  = 0
         for batchIndex, ( images, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            out1, out2, out3 = self( { 'model' : cnn, 'input' : images } )
            loss1 = self.lossFunc( out1, images )
            loss2 = self.lossFunc( out2, images )
            loss3 = self.lossFunc( out3, images )

            # Backward Propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss1.backward( )       # Calculate Gradients
            loss2.backward( )       # Calculate Gradients
            loss3.backward( )       # Calculate Gradients
            optimizer.step( )
            total = total + loss1.item( ) + loss2.item( ) + loss3.item( )

            # Logging
            progress.Update( batchIndex, len( loader ), 'Epoch: {} | Loss: {}'.format( self.epochs + epoch, total ) )

         folder = 'Images/Reconstruct/'
         if( not os.path.exists( folder ) ):
            os.mkdir( folder )
         for i in range( len( images ) ):
            utils.save_image( images[ i ], 'Images/Reconstruct/Epoch{}-{}I.png'.format( epoch, i ) )
            utils.save_image( out1[ i ],   'Images/Reconstruct/Epoch{}-{}D1.png'.format( epoch, i ) )
            utils.save_image( out2[ i ],   'Images/Reconstruct/Epoch{}-{}D2.png'.format( epoch, i ) )
            utils.save_image( out3[ i ],   'Images/Reconstruct/Epoch{}-{}D3.png'.format( epoch, i ) )
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
            out1, out2, out3 = self( { 'model' : cnn, 'input' : images } )
            loss1 = self.lossFunc( out1, images )
            loss2 = self.lossFunc( out2, images )
            loss3 = self.lossFunc( out3, images )
            total = total + loss1.item( ) + loss2.item( ) + loss3.item( )

            # Logging
            progress.Update( batchIndex, len( loader ), 'Epoch: {} | Loss: {}'.format( self.epochs + epoch, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )
   
   def Hook( self, name ):
      def hook( model, input, output ):
         self.activation[ name ] = output.detach( )
      return( hook )
