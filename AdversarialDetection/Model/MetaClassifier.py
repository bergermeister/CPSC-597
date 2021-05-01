import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time
from Utility.ProgressBar import ProgressBar

class MetaClassifier( nn.Module ):
   def __init__( self, cudaEnable ):
      super( ).__init__( )

      # Record cuda enabled flag
      self.cudaEnable = ( cudaEnable == 'True' )
      self.accuracy   = 0
      self.epochs     = 0

      # Linear Layers
      self.ll1 = nn.Linear(  40,   80 )
      self.ll2 = nn.Linear(  80,  160 )
      self.ll3 = nn.Linear( 160,    2 )
      self.act = nn.Sigmoid( )

      # Define Loss Criteria
      self.lossFunc = nn.CrossEntropyLoss( )

      if( self.cudaEnable ):
         self.cuda( )

   def forward( self, x ):
      out = F.relu( self.ll1( x ) )
      out = F.relu( self.ll2( out ) )
      out = self.act( self.ll3( out ) )
      
      return( out )

   def Load( self, path ):
      state = torch.load( path )
      self.load_state_dict( state[ 'model' ] )
      self.accuracy = state[ 'acc' ]
      self.epochs   = state[ 'epoch' ]

   def Train( self, loader, epochs, batch_size ):
      self.train( True )   # Place the model into training mode

      optimizer = torch.optim.Adam( self.parameters( ), lr = 0.0002, weight_decay = 0.00001 )
      progress = ProgressBar( 40, 80 )
      beginTrain = time( )
      print( "Begin Training..." )
      for epoch in range( epochs ):
         beginEpoch = time( )

         trainLoss = 0
         total     = 0
         correct   = 0
         for batchIndex, ( input, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               input, labels = input.cuda( ), labels.cuda( )
            out = self( input )
            loss = self.lossFunc( out, labels )
            
            # Backward propagation
            optimizer.zero_grad( )  # Clear Gradients
            loss.backward( )        # Calculate Gradients
            optimizer.step( )       # Update Weights
            
            # Logging
            _, predicted = out.max( 1 )
            trainLoss    += loss.item( )
            total        += ( labels.size( 0 ) )
            correct      += predicted.eq( labels ).sum( ).item( )
            progress.Update( batchIndex, len( loader ), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( self.epochs + epoch, trainLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Training...' )

   def Test( self, loader, batch_size ):
      self.eval( )         # Place the model into test/evaluation mode
      progress = ProgressBar( 40, 80 )

      testLoss = 0
      total    = 0
      correct  = 0
      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( input, labels ) in enumerate( loader ):
            # Forward pass
            if( self.cudaEnable ):
               input, labels = input.cuda( ), labels.cuda( )
            out = self( input )
            loss = self.lossFunc( out, labels )

            # Logging
            _, predicted = out.max( 1 )
            testLoss     += loss.item( )
            total        += ( labels.size( 0 ) )
            correct      += predicted.eq( labels ).sum( ).item( )
            progress.Update( batchIndex, len( loader ), '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % ( testLoss / ( batchIndex + 1 ), 100. * correct / total, correct, total ) )
      print( 'End Evaluation...' )

      return( 100. * correct / total )