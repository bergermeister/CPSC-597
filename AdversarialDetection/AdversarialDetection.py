## @package AdversarialDetection
# 
import sys
import os
import argparse
import torch
#import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
from PIL import Image as im 
from Model.CNN import CNN
from Model.Adversary import Adversary
from Utility.Data import Data

##
# @brief
# Application Entry
#
# @details
# @par
# This method is the application entry point.
def Main( ):
   print( "Parsing Arguments" )
   parser = argparse.ArgumentParser( description = "CPSC-597" )
   parser.add_argument( '--dataset',    type = str, default = 'mnist',     choices = [ 'mnist' ] )
   parser.add_argument( '--data_path',  type = str, required = True,       help = 'path to dataset' )
   parser.add_argument( '--batch_size', type = int, default = 64,          help = 'Size of a batch' )
   parser.add_argument( '--mode',       type = str, default = 'train',     choices = [ 'train', 'test', 'adversary' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,          help = 'The number of epochs to run')
   parser.add_argument( '--cuda',       type = str, default = 'True',      help = 'Availability of cuda' )
   parser.add_argument( '--cnn',        type = str, default = 'cnn.state', help = 'Path to CNN Model State' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Model" )
   model = CNN( 1, args.cuda )

   if( os.path.exists( args.cnn ) ):
      print( "Loading model state" )
      state = torch.load( args.cnn )
      model.load_state_dict( state[ 'model' ] )
      model.accuracy = state[ 'acc' ]
      model.epochs   = state[ 'epoch' ]

   if( args.mode == 'train' ):
      model.Train( data.train_loader, args.epochs, args.batch_size )
      acc = model.Test( data.test_loader, args.batch_size )
      if( acc > model.accuracy ):
         Save( model, acc, model.epochs + args.epochs, args.cnn )
   elif( args.mode == 'test' ):
      acc = model.Test( data.test_loader, args.batch_size )
   elif( args.mode == 'adversary' ):
      epsilons = [ .1 ] #[ 0, .05, .1, .15, .2, .25, .3 ]
      adversary = Adversary( model )

      ######################################################################
      # Run Attack
      # ~~~~~~~~~~
      # 
      # The last part of the implementation is to actually run the attack. Here,
      # we run a full test step for each epsilon value in the *epsilons* input.
      # For each epsilon we also save the final accuracy and some successful
      # adversarial examples to be plotted in the coming sections. Notice how
      # the printed accuracies decrease as the epsilon value increases. Also,
      # note the :math:`\epsilon=0` case represents the original test accuracy,
      # with no attack.
      # 
      accuracies = []
      examples = []

      # Run test for each epsilon
      for eps in epsilons:
         acc, ex = adversary.Attack( data.test_loader, eps )
         accuracies.append( acc )
         examples.append( ex )
         for i in range( len( ex ) ):
            example = im.fromarray( ( ex[ i ][ 2 ] * 255 ).astype( 'uint8' ) )
            example.save( 'Images/Example{}From{}To{}.png'.format( i, ex[ i ][ 0 ], ex[ i ][ 1 ] ) )
            for j in range( len( ex[ i ][ 3 ] ) ):
               example = ex[ i ][ 3 ][ j ][ 0 ][ 0 ]
               example = example.cpu()
               example = example.numpy()
               example = im.fromarray( ( example * 255 ).astype( 'uint8' ) )
               example.save( 'Images/Example{}Layer{}.png'.format( i, j ) )

##
# @brief
# Save Model State
#
# @details
# This method saves the given model state as well as records the accuracy and number epochs run to the path provided.
#
# @param model    CNN Model
# @param acc      Model evaluation accuracy
# @param epoch    Number of epochs performed
# @param path     File path
def Save( model, acc, epoch, path ):
   print( 'Saving...' )
   state = {
      'model': model.state_dict( ),
      'acc': acc,
      'epoch': epoch,
   }
   torch.save( state, path )
   print( 'Save complete.' )

if __name__ == "__main__":
   sys.exit( int( Main( ) or 0 ) )
   
