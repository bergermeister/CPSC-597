##
# @package Encoder CNN
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
class Encoder( nn.Module ):
   def __init__( self, channels, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.channels = channels
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 100000
      self.epochs     = 0
      self.indices    = {}
      self.activation = {}

      # Convolutional Layers
      self.cl1 = nn.Conv2d( self.channels, 16, 5, 1, 1 ) # Convolutional Layer 1 (16, 16, 16)
      self.cl2 = nn.Conv2d(            16, 32, 5, 1, 1 ) # Convolutional Layer 2 (32,  8 , 8)
      self.cl3 = nn.Conv2d(            32, 64, 5, 1, 1 ) # Convolutional Layer 3 (64,  4,  4)
      self.mp  = nn.MaxPool2d( 2, stride = 2, return_indices = True )
      
      # Deconvolutional Layers
      self.dl1 = nn.ConvTranspose2d( 64,            32, 5, 1, 1 )
      self.dl2 = nn.ConvTranspose2d( 32,            16, 4, 1, 0 )
      self.dl3 = nn.ConvTranspose2d( 16, self.channels, 5, 1, 1 )
      self.up   = nn.MaxUnpool2d( 2, 2 )
      self.relu = nn.ReLU( True )
      self.sigm = nn.Sigmoid( )

      # Define Loss Criteria
      self.lossFunc = nn.MSELoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      encoded = self.Encode( x )
      decoded = self.Decode( encoded )
      return( encoded, decoded )

   def Encode( self, x ):
      out, self.indices[ 'mp1' ] = self.mp( self.relu( self.cl1(   x ) ) )
      out, self.indices[ 'mp2' ] = self.mp( self.relu( self.cl2( out ) ) )
      out, self.indices[ 'mp3' ] = self.mp( self.relu( self.cl3( out ) ) )
      return( out )

   def Decode( self, x ):
      out = self.relu( self.dl1( self.up(   x, self.indices[ 'mp3' ] ) ) )
      out = self.relu( self.dl2( self.up( out, self.indices[ 'mp2' ] ) ) )
      out = self.sigm( self.dl3( self.up( out, self.indices[ 'mp1' ] ) ) )
      return( out )

   def Load( self, path ):
      state = torch.load( path )
      self.load_state_dict( state[ 'model' ] )
      self.accuracy = state[ 'acc' ]
      self.epochs   = state[ 'epoch' ]

   def Train( self, loader, epochs, batch_size ):
      self.train( True ) # Place the model into training mode

      progress = ProgressBar( 40, 80 )
      total = 0
      beginTrain = time( )
      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )

      print( "Begin Training..." )
      for epoch in range( epochs ):
         beginEpoch = time( )
         
         images = 0
         encoded = 0
         decoded = 0
         total = 0
         for batchIndex, ( images, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            encoded, decoded = self( images )
            loss = self.lossFunc( decoded, images )

            # Backward propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights
            total = total + loss.item()

            # Logging
            progress.Update( batchIndex, len( loader ), 'Epoch: {} | Loss: {}'.format( self.epochs + epoch, total ) )

         folder = 'Images/AuxDecode/'
         if( not os.path.exists( folder ) ):
            os.mkdir( folder )
         for i in range( len( images ) ):
            utils.save_image( images[ i ],  'Images/AuxDecode/Epoch{}-{}I.png'.format( epoch, i ) )
            utils.save_image( decoded[ i ], 'Images/AuxDecode/Epoch{}-{}D.png'.format( epoch, i ) )

      print( 'End Training...' )

   def Test( self, loader, batch_size ):
      self.eval( ) # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )
      total = 0
      images = 0
      encoded = 0
      decoded = 0

      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         total = 0
         for batchIndex, ( images, labels ) in enumerate( loader ):
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            encoded, decoded = self( images )
            loss    = self.lossFunc( decoded, images )
            total   = total + loss.item( )

            progress.Update( batchIndex, len( loader ), 'Batch: {} | Loss: {}'.format( batchIndex, total ) )

      print( 'End Evaluation...' )

      return( total )
   
   def Reconstruct( self, x ):
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
      out1 = self.sigm( self.dl3( self.activation[ 'cl1' ] ) )

      # Decode second convolutional layer
      out2, indices = self.mp( self.relu( self.activation[ 'cl2' ] ) )
      out2 = self.up( out2, indices )
      out2 = self.relu( self.dl2( out2 ) )
      out2 = self.up( out2, cnn.indices[ 'mp1' ] )
      out2 = self.sigm( self.dl3( out2 ) )

      # Decode third convolutional layer
      out3 = self.relu( self.dl1( self.activation[ 'cl3' ] ) )
      out3 = self.up( out3, cnn.indices[ 'mp2' ] )
      out3 = self.relu( self.dl2( out3 ) )
      out3 = self.up( out3, cnn.indices[ 'mp1' ] )
      out3 = self.sigm( self.dl3( out3 ) )

      return( out1, out2, out3 )

   def Hook( self, name ):
      def hook( model, input, output ):
         self.activation[ name ] = output.detach( )
      return( hook )
