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

activation = {}

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
   parser.add_argument( '--epsilon',    type = float, default = 0.1,       help = 'Perturbed image pixel adjustment factor' )
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
      #model.cl1.register_forward_hook( get_activation( 'cl1' ) )
      epsilon = args.epsilon
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

      acc, ex = adversary.Attack( data.test_loader, epsilon )
      accuracies.append( acc )
      examples.append( ex )
      for i in range( len( ex ) ):
         init_pred = ex[ i ][ 0 ]
         final_pred = ex[ i ][ 1 ]
         example    = ex[ i ][ 2 ]
         init_act   = ex[ i ][ 3 ]
         final_act  = ex[ i ][ 4 ]
         example = im.fromarray( ( ex[ i ][ 2 ] * 255 ).astype( 'uint8' ) )
         example.save( 'Images/Eps{}Example{}From{}To{}.png'.format( epsilon, i, init_pred, final_pred ) )
         for j in range( init_act.size( 0 ) ):
            example = init_act[ j ]
            example = example.cpu()
            example = example.numpy()
            example = im.fromarray( ( example * 255 ).astype( 'uint8' ) )
            example.save( 'Images/Eps{}Example{}From{}To{}Act{}Init.png'.format( epsilon, i, init_pred, final_pred, j ) )
         for j in range( final_act.size( 0 ) ):
            example = final_act[ j ]
            example = example.cpu()
            example = example.numpy()
            example = im.fromarray( ( example * 255 ).astype( 'uint8' ) )
            example.save( 'Images/Eps{}Example{}From{}To{}Act{}Final.png'.format( epsilon, i, init_pred, final_pred, j ) )

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

def get_activation( name ):
   def hook( model, input, output ):
      activation[ name ] = output.detach( )
   return( hook )

if __name__ == "__main__":
   sys.exit( int( Main( ) or 0 ) )
   
