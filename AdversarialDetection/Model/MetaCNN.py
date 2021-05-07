import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time
from Utility.ProgressBar import ProgressBar
from Model.CNN import CNN
from Model.Reconstructor import Reconstructor
from Model.Detector import Detector

class MetaCNN( nn.Module ):
   def __init__( self, channels, args ):
      super( ).__init__( )

      self.cudaEnable = args.cuda

      self.cnn = CNN( channels, args.cuda )
      self.rec = Reconstructor( channels, args.cuda )
      self.det = Detector( channels * 4, args.cuda )

      if( os.path.exists( args.cnn ) ):
         print( "Meta CNN: Loading CNN state" )
         self.cnn.Load( args.cnn )
      if( os.path.exists( args.recon ) ):
         print( "Meta CNN: Loading Reconstructor state" )
         self.rec.Load( args.recon )
      if( os.path.exists( args.detect ) ):
         print( "Meta CNN: Loading Detector state" )
         self.det.Load( args.detect )

   def forward( self, x ):
      with torch.no_grad():
         x1, x2, x3 = self.rec( { 'model' : self.cnn, 'input' : x } )   # Reconstruct images after cl1, cl2, and cl3
         xM = torch.cat( ( x, x1, x2, x3 ), 1 )                         # Concatenate x, x1, x2, and x3
         _, adv = self.det( xM ).max( 1 )                               # Detect Adversarial images
         _, y   = self.cnn( x ).max( 1 )                                # Forward pass standard CNN
         out = torch.zeros( adv.size( 0 ), dtype = torch.long, device = 'cuda' )
         for i in range( adv.size( 0 ) ):
            if( adv[ i ] == 1 ):
               out[ i ] = 10
            else:
               out[ i ] = y[ i ]
      return( out )

   def Test( self, loader, batch_size ):
      self.cnn.eval( )  # Place the model into test/evaluation mode
      self.rec.eval( )  # Place the model into test/evaluation mode
      self.det.eval( )  # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )

      total    = 0
      correct  = 0
      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( input, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               input, labels = input.cuda( ), labels.cuda( )
            out = self( input )

            # Logging
            total        += ( labels.size( 0 ) )
            correct      += out.eq( labels ).sum( ).item( )
            progress.Update( batchIndex, len( loader ), '| Acc: %.3f%% (%d/%d)'
                             % ( 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )