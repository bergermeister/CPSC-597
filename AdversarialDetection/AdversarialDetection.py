## @package AdversarialDetection
# 
import sys
import os
import argparse
import numpy as np
import torch
from torchvision import utils
from Model.CNN import CNN
from Model.Reconstruct import Reconstruct
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
   parser.add_argument( '--mode',       type = str, default = 'train',           choices = [ 'encoder', 'classifier', 'train', 'test', 'recon', 'adversary' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,                help = 'The number of epochs to run')
   parser.add_argument( '--epsilon',    type = float, default = 0.1,             help = 'Perturbed image pixel adjustment factor' )
   parser.add_argument( '--cuda',       type = str, default = 'True',            help = 'Availability of cuda' )
   parser.add_argument( '--cnn',        type = str, default = 'cnn.state',       help = 'Path to CNN Model State' )
   parser.add_argument( '--recon',      type = str, default = 'recon.state',     help = 'Path to Reconstruction Model State' )
   parser.add_argument( '--encoder',    type = str, default = 'encoder.state',   help = 'Path to Encoder Model State' )
   parser.add_argument( '--classifier', type = str, default = 'clasifier.state', help = 'Path to Classifier Model State' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Models" )
   cnn        = CNN( data.channels, args.cuda )
   recon      = Reconstruct( data.channels, args.cuda )
   encoder    = Encoder( data.channels, args.cuda )
   classifier = Classifier( encoder, args.cuda )

   if( os.path.exists( args.cnn ) ):
      print( "Loading CNN state" )
      cnn.Load( args.cnn )
   if( os.path.exists( args.recon ) ):
      print( "Loading CNN state" )
      recon.Load( args.recon )
   if( os.path.exists( args.encoder ) ):
      print( "Loading encoder state" )
      encoder.Load( args.encoder )
   if( os.path.exists( args.classifier ) ):
      print( "Loading classifier state" )
      classifier.Load( args.classifier )

   if( args.mode == 'train' ):
      cnn.Train( data.train_loader, args.epochs, args.batch_size )
      acc = cnn.Test( data.test_loader, args.batch_size )
      if( acc > cnn.accuracy ):
         Save( cnn, 0, cnn.epochs + args.epochs, args.cnn )
   elif( args.mode == 'test' ):
      cnn.eval()
      acc = cnn.Test( data.test_loader, args.batch_size )
   elif( args.mode == 'recon' ):
      recon.Train( cnn, data.train_loader, args.epochs, args.batch_size )
      Save( recon, 0, recon.epochs + recon.epochs, args.recon )
   elif( args.mode == 'encoder' ):
      encoder.Train( data.train_loader, args.epochs, args.batch_size )
      loss = encoder.Test( data.test_loader, args.batch_size )
      Save( encoder, loss, encoder.epochs + args.epochs, args.encoder )
   elif( args.mode == 'classifier' ):
      classifier.Train( data.train_loader, args.epochs, args.batch_size )
      acc = classifier.Test( data.test_loader, args.batch_size )
      if( acc > classifier.accuracy ):
         Save( classifier, 0, classifier.epochs + args.epochs, args.classifier )
   elif( args.mode == 'adversary' ):
      labelStr = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
      cnn.eval( )
      recon.eval( )
      accuracies = []

      epsilon = args.epsilon
      adversary = Adversary( cnn )
      
      print( "Generating Adversarial Images" )
      advloader, results = adversary.PGDAttack( "cuda", data.test_loader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      print( "Evaluating Normal Images" )
      accuracies.append( cnn.Test( data.test_loader, args.batch_size ) )
      print( "Evaluating Adversarial Images" )
      accuracies.append( cnn.Test( advloader, args.batch_size ) )

      for batchIndex in range( len( results ) ):
         batch = results[ batchIndex ]
         if( cnn.cudaEnable ):
            outputOrig = cnn( batch[ 'orig' ].cuda() )
            outputAdvs = cnn( batch[ 'adv'  ].cuda() )
         else:
            outputOrig = cnn( batch[ 'orig' ] )
            outputAdvs = cnn( batch[ 'adv'  ] )
         
         predOrig = outputOrig.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
         predAdvs = outputAdvs.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
         for i in range( len( batch[ 'status' ] ) ):
            if( ( predOrig[ i ].item( ) == batch[ 'label' ][ i ] ) and ( batch[ 'status' ][ i ] == True ) ):
               if( cnn.cudaEnable ):
                  orig1, orig2, orig3 = recon( { 'model' : cnn, 'input' : batch[ 'orig' ][ i ].unsqueeze( 0 ).cuda( ) } )
                  advs1, advs2, advs3 = recon( { 'model' : cnn, 'input' : batch[ 'adv'  ][ i ].unsqueeze( 0 ).cuda( ) } )
               else:
                  orig1, orig2, orig3 = recon( { 'model' : cnn, 'input' : batch[ 'orig' ][ i ].unsqueeze( 0 ) } )
                  advs1, advs2, advs3 = recon( { 'model' : cnn, 'input' : batch[ 'adv'  ][ i ].unsqueeze( 0 ) } )
               
               folder = 'Images/Epsilon{}Batch{}Example{}/'.format( epsilon, batchIndex, i )
               if( os.path.exists( folder ) ):
                  for filename in os.listdir( folder ):
                     filePath = os.path.join( folder, filename )
                     try:
                        if( os.path.isfile( filePath ) or os.path.islink( filePath ) ):
                           os.unlink( filePath )
                     except Exception as e:
                        print( 'Failed to delete {} | Reasion: {}'.format( filePath, e ) )
               else:
                  os.mkdir( folder )
               utils.save_image( orig1, os.path.join( folder, '{}-{}-{}_CL{}O.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 1 ) ) )
               utils.save_image( orig2, os.path.join( folder, '{}-{}-{}_CL{}O.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 2 ) ) )
               utils.save_image( orig3, os.path.join( folder, '{}-{}-{}_CL{}O.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 3 ) ) )
               utils.save_image( advs1, os.path.join( folder, '{}-{}-{}_CL{}A.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 1 ) ) )
               utils.save_image( advs2, os.path.join( folder, '{}-{}-{}_CL{}A.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 2 ) ) )
               utils.save_image( advs3, os.path.join( folder, '{}-{}-{}_CL{}A.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ], 3 ) ) )

               utils.save_image( batch[ 'orig' ][ i ], os.path.join( folder, '{}-{}-{}A.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ] ) ) )
               utils.save_image( batch[ 'adv'  ][ i ], os.path.join( folder, '{}-{}-{}O.png'.format( labelStr[ batch[ 'label' ][ i ] ], labelStr[ predOrig[ i ].item( ) ], labelStr[ predAdvs[ i ].item( ) ] ) ) )


      #for batchIndex, ( images, labels ) in enumerate( advloader ):
      #   if( cnn.cudaEnable ):
      #      images = images.cuda( )
      #   outputs = cnn( images )
      #   final_pred = outputs.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
      #   for i in range( len( images ) ):
      #      if( success[ batchIndex ][ i ] == True ):
      #         input = images[ i ].unsqueeze( 0 )
      #         out1, out2, out3 = recon( { 'model' : cnn, 'input' : input } )
      #
      #         utils.save_image( out1, 'Images/Epsilon{}Batch{}Example{}_{}-{}CL{}.png'.format( epsilon, batchIndex, i, labelStr[ labels[ i ] ], labelStr[ final_pred[ i ].item( ) ], 1 ) )
      #         utils.save_image( out2, 'Images/Epsilon{}Batch{}Example{}_{}-{}CL{}.png'.format( epsilon, batchIndex, i, labelStr[ labels[ i ] ], labelStr[ final_pred[ i ].item( ) ], 2 ) )
      #         utils.save_image( out3, 'Images/Epsilon{}Batch{}Example{}_{}-{}CL{}.png'.format( epsilon, batchIndex, i, labelStr[ labels[ i ] ], labelStr[ final_pred[ i ].item( ) ], 3 ) )
      #
      #         example = images[ i ]
      #         example = example.cpu()
      #         utils.save_image( example, 'Images/Epsilon{}Batch{}Example{}-{}.png'.format( epsilon, batchIndex, i, success[ batchIndex ][ i ] ) )
      #   break

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
   
