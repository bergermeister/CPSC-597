## @package AdversarialDetection
# 
import sys
import os
import argparse
import torch
from Model.CNN import CNN
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
   parser.add_argument( '--mode',       type = str, default = 'train',     choices = [ 'train', 'test' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,          help = 'The number of epochs to run')
   parser.add_argument( '--cuda',       type = str, default = True,        help = 'Availability of cuda' )
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
         save( model, acc, model.epochs + args.epochs, args.cnn )
   elif( args.mode == 'test' ):
      acc = model.Test( data.test_loader, args.batch_size )

##
# @brief
# Save Model
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
   
