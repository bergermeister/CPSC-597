## @package AdversarialDetection
# 
import sys
import os
import argparse
import torch
import numpy as np
from torchvision import utils
from torch.nn import functional as F
from PIL import Image as im 
from Model.CNN import CNN
from Model.Encoder import Encoder
from Model.Classifier import Classifier
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
   parser.add_argument( '--dataset',    type = str, default = 'mnist',           choices = [ 'mnist', 'cifar10' ] )
   parser.add_argument( '--data_path',  type = str, required = True,             help = 'path to dataset' )
   parser.add_argument( '--batch_size', type = int, default = 64,                help = 'Size of a batch' )
   parser.add_argument( '--mode',       type = str, default = 'train',           choices = [ 'encoder', 'classifier', 'train', 'test', 'adversary' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,                help = 'The number of epochs to run')
   parser.add_argument( '--epsilon',    type = float, default = 0.1,             help = 'Perturbed image pixel adjustment factor' )
   parser.add_argument( '--cuda',       type = str, default = 'True',            help = 'Availability of cuda' )
   parser.add_argument( '--cnn',        type = str, default = 'cnn.state',       help = 'Path to CNN Model State' )
   parser.add_argument( '--encoder',    type = str, default = 'encoder.state',   help = 'Path to Encoder Model State' )
   parser.add_argument( '--classifier', type = str, default = 'clasifier.state', help = 'Path to Classifier Model State' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Models" )
   cnn        = CNN( data.channels, args.cuda )
   encoder    = Encoder( data.channels, args.cuda )
   classifier = Classifier( encoder, args.cuda )

   if( os.path.exists( args.cnn ) ):
      print( "Loading CNN state" )
      state = torch.load( args.cnn )
      cnn.load_state_dict( state[ 'model' ] )
      cnn.accuracy = state[ 'acc' ]
      cnn.epochs   = state[ 'epoch' ]
   if( os.path.exists( args.encoder ) ):
      print( "Loading encoder state" )
      state = torch.load( args.encoder )
      encoder.load_state_dict( state[ 'model' ] )
      encoder.accuracy = state[ 'acc' ]
      encoder.epochs   = state[ 'epoch' ]
   if( os.path.exists( args.classifier ) ):
      print( "Loading classifier state" )
      state = torch.load( args.classifier )
      classifier.load_state_dict( state[ 'model' ] )
      classifier.accuracy = state[ 'acc' ]
      classifier.epochs   = state[ 'epoch' ]

   if( args.mode == 'train' ):
      cnn.Train( data.train_loader, args.epochs, args.batch_size )
      acc = cnn.Test( data.test_loader, args.batch_size )
      if( acc > cnn.accuracy ):
         Save( cnn, 0, cnn.epochs + args.epochs, args.cnn )
   if( args.mode == 'encoder' ):
      encoder.Train( data.train_loader, args.epochs, args.batch_size )
      loss = encoder.Test( data.test_loader, args.batch_size )
      Save( encoder, loss, encoder.epochs + args.epochs, args.encoder )
   elif( args.mode == 'classifier' ):
      classifier.Train( data.train_loader, args.epochs, args.batch_size )
      acc = classifier.Test( data.test_loader, args.batch_size )
      if( acc > classifier.accuracy ):
         Save( classifier, 0, classifier.epochs + args.epochs, args.classifier )
   elif( args.mode == 'test' ):
      #acc = classifier.Test( data.test_loader, args.batch_size )
      encoder.Test( data.test_loader, args.batch_size )
   elif( args.mode == 'adversary' ):
      cnn.eval( )
      encoder.eval( )
      classifier.eval( )

      model = classifier

      epsilon = args.epsilon
      adversary = Adversary( model )

      accuracies = []
      examples = []

      #accuracies.append( classifier.Test( data.test_loader, args.batch_size ) )
      print( "Generating Adversarial Images" )
      advloader, success = adversary.PGDAttack( "cuda", data.test_loader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      accuracies.append( model.Test( advloader, args.batch_size ) )

      cnn.cl1.register_forward_hook( Deconvolution( 'cl1', encoder ) )
      cnn.cl2.register_forward_hook( Deconvolution( 'cl2', encoder ) )
      cnn.cl3.register_forward_hook( Deconvolution( 'cl3', encoder ) )
      classifier.encoder.cl1.register_forward_hook( Deconvolution( 'cl1', encoder ) )
      classifier.encoder.cl2.register_forward_hook( Deconvolution( 'cl2', encoder ) )
      classifier.encoder.cl3.register_forward_hook( Deconvolution( 'cl3', encoder ) )
      for batchIndex, ( images, labels ) in enumerate( advloader ):
         if( model.cudaEnable ):
            images = images.cuda( )
         outputs = model( images )
         for i in range( len( images ) ):
            #if( success[ batchIndex ][ i ] == True ):
            input = images[ i ].unsqueeze( 0 )
            model( input )
            for index, (key, value) in enumerate( activation.items( ) ):
               example = value.cpu()
               example = ( example - example.min() ) / example.max()
               utils.save_image( example, 'Images/Epsilon{}Batch{}Example{}CL{}.png'.format( epsilon, batchIndex, i, index + 1 ) )

            encoded, decoded = encoder( input )
            utils.save_image( decoded, 'Images/Epsilon{}Batch{}Example{}-{}-Decoded.png'.format( epsilon, batchIndex, i, success[ batchIndex ][ i ] ) )

            example = images[ i ]
            example = example.cpu()
            utils.save_image( example, 'Images/Epsilon{}Batch{}Example{}-{}.png'.format( epsilon, batchIndex, i, success[ batchIndex ][ i ] ) )
         break

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

def Deconvolution( name, encoder ):
   def hook( model, input, output ):
      if( name == 'cl1' ):
         activation[ name ] = encoder.act( encoder.dl1( F.relu( output ) ) ).detach( )
      elif( name == 'cl2' ):
         activation[ name ] = encoder.act( encoder.dl1( F.relu( encoder.dl2( F.relu( output ) ) ) ) ).detach( )
         #activation[ name ] = encoder.dl1( encoder.dl2( output ) ).detach( )
      elif( name == 'cl3' ):
         activation[ name ] = encoder.act( encoder.dl1( F.relu( encoder.dl2( F.relu( encoder.dl3( F.relu( output ) ) ) ) ) ) ).detach( )      
         #activation[ name ] = encoder.dl1( encoder.dl2( encoder.dl3( output ) ) ).detach( )      
   return( hook )

if __name__ == "__main__":
   sys.exit( int( Main( ) or 0 ) )
   
