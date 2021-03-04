##
# @package Encoder CNN
#
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

      # Convolutional Layers
      self.cl1 = nn.Conv2d( self.channels, 12, 4, 2, 1 ) # Convolutional Layer 1 (12, 16, 16)
      self.cl2 = nn.Conv2d( 12, 24, 4, 2, 1 )            # Convolutional Layer 2 (24,  8 , 8)
      self.cl3 = nn.Conv2d( 24, 48, 4, 2, 1 )            # Convolutional Layer 3 (48,  4,  4)

      # Deconvolutional Layers
      self.dl1 = nn.ConvTranspose2d( 12, self.channels, 4, 2, 1 )
      self.dl2 = nn.ConvTranspose2d( 24, 12, 4, 2, 1 )
      self.dl3 = nn.ConvTranspose2d( 48, 24, 4, 2, 1 )
      self.act = nn.Sigmoid( );

      # Define Loss Criteria
      self.loss = nn.BCELoss( )

      if( self.cudaEnable ):
         self.cuda( )
            
   def forward( self, x ):
      encoded = self.Encode( x )
      decoded = self.Decode( encoded )
      return( encoded, decoded )

   def Encode( self, x ):
      out = F.relu( self.cl1( x ) )
      out = F.relu( self.cl2( out ) )
      out = F.relu( self.cl3( out ) )
      return( out )

   def Decode( self, x ):
      out = F.relu( self.dl3( x ) )
      out = F.relu( self.dl2( out ) )
      out = self.dl1( out )
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
      total = 0
      beginTrain = time( )
      optimizer = torch.optim.Adam( self.parameters( ) ) #, lr = 0.0002, weight_decay = 0.00001 )

      images = 0
      encoded = 0
      decoded = 0

      print( "Begin Training..." )
      for epoch in range( epochs ):
         beginEpoch = time( )
         total = 0
         for batchIndex, ( images, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               images, labels = images.cuda( ), labels.cuda( )
            encoded, decoded = self( images )
            loss = self.loss( decoded, images )

            # Backward propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights
            
            # Logging
            total = total + loss.item()
            progress.Update( batchIndex, len( loader ), 'Epoch: {} | Loss: {}'.format( self.epochs + epoch, total ) )

         for i in range( len( images ) ):
            utils.save_image( images[ i ], 'Images/Epoch{}-{}I.png'.format( epoch, i ) )
            #utils.save_image( encoded[ i ].reshape( 3, 16, 16 ), 'Images/Epoch{}-{}E.png'.format( epoch, i ) )
            utils.save_image( decoded[ i ], 'Images/Epoch{}-{}D.png'.format( epoch, i ) )

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
            loss    = self.loss( decoded, images )
            total   = total + loss.item( )

            progress.Update( batchIndex, len( loader ), 'Batch: {} | Loss: {}'.format( batchIndex, total ) )

            for i in range( len( images ) ):
               utils.save_image( images[ i ], 'Images/Batch{}-{}I.png'.format( batchIndex, i ) )
               #utils.save_image( encoded[ i ].reshape( 3, 16, 16 ), 'Images/Batch{}-{}E.png'.format( batch_size, i ) )
               utils.save_image( decoded[ i ], 'Images/Batch{}-{}D.png'.format( batchIndex, i ) )

      print( 'End Evaluation...' )

      return( total )
   
