## @package AdversarialDetection
# 
import sys
import os
import argparse
import numpy as np
import torch
import csv
from torchvision import utils
from Model.CNN import CNN
from Model.Reconstructor import Reconstructor
from Model.Detector import Detector
from Model.Adversary import Adversary
from Model.MetaCNN import MetaCNN

from Utility.Data import Data
from Utility.ProgressBar import ProgressBar

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
   parser.add_argument( '--dataset',    type = str, default = 'mnist',        choices = [ 'mnist', 'cifar10' ] )
   parser.add_argument( '--data_path',  type = str, required = True,          help = 'Path to dataset' )
   parser.add_argument( '--batch_size', type = int, default = 64,             help = 'Size of a batch' )
   parser.add_argument( '--mode',       type = str, default = 'cnn',          choices = [ 'cnn', 'recon', 'detect', 'test', 'adversary' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,             help = 'The number of epochs to run')
   parser.add_argument( '--epsilon',    type = float, default = 0.1,          help = 'Perturbed image pixel adjustment factor' )
   parser.add_argument( '--cuda',       type = str, default = 'True',         help = 'Availability of cuda' )
   parser.add_argument( '--cnn',        type = str, default = 'cnn.state',    help = 'Path to CNN Model State' )
   parser.add_argument( '--recon',      type = str, default = 'recon.state',  help = 'Path to Reconstructor Model State' )
   parser.add_argument( '--detect',     type = str, default = 'detect.state', help = 'Path to Detector Model State' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Models" )
   cnn    = CNN( data.channels, args.cuda )
   recon  = Reconstructor( data.channels, args.cuda )
   detect = Detector( data.channels * 4, args.cuda )
   meta   = MetaCNN( data.channels, args )

   if( os.path.exists( args.cnn ) ):
      print( "Loading CNN state" )
      cnn.Load( args.cnn )
   if( os.path.exists( args.recon ) ):
      print( "Loading Reconstructor state" )
      recon.Load( args.recon )
   if( os.path.exists( args.detect ) ):
      print( "Loading Detector state" )
      detect.Load( args.detect )

   adversary  = Adversary( cnn )

   if( args.mode == 'cnn' ):
      cnn.Train( data.train_loader, args.epochs, args.batch_size )
      acc = cnn.Test( data.test_loader, args.batch_size )
      if( acc > cnn.accuracy ):
         Save( cnn, acc, cnn.epochs + args.epochs, args.cnn )
   elif( args.mode == 'recon' ):
      recon.Train( cnn, data.train_loader, args.epochs, args.batch_size )
      Save( recon, 0, recon.epochs + recon.epochs, args.recon )
   elif( args.mode == 'detect' ):
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.train_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, args.epsilon, args.epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateDetectorLoader( cnn, recon, successLoader, advloader )
      detect.Train( metaloader, args.epochs, args.batch_size )

      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, args.epsilon, args.epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateDetectorLoader( cnn, recon, data.test_loader, advloader )
      acc = detect.Test( metaloader, args.batch_size )
      if( acc > detect.accuracy ):
         Save( detect, acc, detect.epochs + args.epochs, args.detect )
   elif( args.mode == 'test' ):
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, args.epsilon, args.epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateMetaLoader( data.test_loader, advloader )
      acc = meta.Test( metaloader, args.batch_size )

   elif( args.mode == 'adversary' ):
      labelStr = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
      cnn.eval( )
      recon.eval( )
      accuracies = []
      
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      accuracies.append( cnn.Test( successLoader, args.batch_size ) )
      print( "Generating Adversarial Images" )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, args.epsilon, args.epsilon / 10.0, 10, -1, 1, False )
      print( "Evaluating Adversarial Images" )
      accuracies.append( cnn.Test( advloader, args.batch_size ) )

      print( "Saving Adversarial Image Examples" )
      #            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
      examples = [ ]
      for i in range( len( labelStr ) ):
         examples.append( { 'total' : 0, 'cl1' : 0, 'cl2' : 0, 'cl3' : 0 } )
      
      progress = ProgressBar( 40, 80 )
      for batchIndex in range( len( results ) ):
         labelBatch     = results[ batchIndex ][ 'label' ]         
         originalBatch  = results[ batchIndex ][ 'orig' ]
         originalOut    = cnn( originalBatch.cuda( ) )
         originalPred   = originalOut.max( 1, keepdim = True )[ 1 ]
         adversaryBatch = results[ batchIndex ][ 'adv' ]
         adversaryOut   = cnn( adversaryBatch.cuda( ) )
         adversaryPred  = adversaryOut.max( 1, keepdim = True )[ 1 ]
         
         for i in range( len( adversaryOut ) ):
            # If the Adversary prediction does not equal the correct label and the example count for the label is less than 100 
            if( ( adversaryPred[ i ] != labelBatch[ i ] ) and ( examples[ labelBatch[ i ] ][ 'total' ] < 100 ) ):
               examples[ labelBatch[ i ] ][ 'total' ] += 1
      
               # Obtain the reconstructed image for the original and adversarial inputs after convolutional layers 1, 2, and 3
               orig1, orig2, orig3 = recon( { 'model' : cnn, 'input' : originalBatch[ i ].unsqueeze( 0 ).cuda( ) } )
               advs1, advs2, advs3 = recon( { 'model' : cnn, 'input' : adversaryBatch[ i ].unsqueeze( 0 ).cuda( ) } )

               # Obtain the classification for the original and adversarial inputs after convolutional layers 1, 2, and 3
               corig1, corig2, corig3 = classifier( { 'model' : cnn, 'input' : originalBatch[ i ].unsqueeze( 0 ).cuda( ) } )
               cadvs1, cadvs2, cadvs3 = classifier( { 'model' : cnn, 'input' : adversaryBatch[ i ].unsqueeze( 0 ).cuda( ) } )
      
               if( cadvs1.max( 1, keepdim = True )[ 1 ] != corig1.max( 1, keepdim = True )[ 1 ] ):
                  examples[ labelBatch[ i ] ][ 'cl1' ] += 1
               elif( cadvs2.max( 1, keepdim = True )[ 1 ] != corig2.max( 1, keepdim = True )[ 1 ] ):
                  examples[ labelBatch[ i ] ][ 'cl2' ] += 1
               elif( cadvs3.max( 1, keepdim = True )[ 1 ] != corig3.max( 1, keepdim = True )[ 1 ] ):
                  examples[ labelBatch[ i ] ][ 'cl3' ] += 1
      
               # Ensure folder exists to store example images
               folder = 'Images/Epsilon{}/'.format( args.epsilon )
               if( not os.path.exists( folder ) ):
                  os.mkdir( folder )
               folder = os.path.join( folder, labelStr[ labelBatch[ i ] ] )
               if( not os.path.exists( folder ) ):
                  os.mkdir( folder )
      
               # Save the original, adversarial, reconstructed, and delta images
               utils.save_image( originalBatch[ i ],  os.path.join( folder,        'Example{}-O.png'.format( examples[ labelBatch[ i ] ][ 'total' ] ) ) )
               utils.save_image( adversaryBatch[ i ], os.path.join( folder,        'Example{}-A.png'.format( examples[ labelBatch[ i ] ][ 'total' ] ) ) )
               utils.save_image( orig1,               os.path.join( folder, 'Example{}-O-CL1-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ corig1.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( orig2,               os.path.join( folder, 'Example{}-O-CL2-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ corig2.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( orig3,               os.path.join( folder, 'Example{}-O-CL3-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ corig3.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( advs1,               os.path.join( folder, 'Example{}-A-CL1-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ cadvs1.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( advs2,               os.path.join( folder, 'Example{}-A-CL2-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ cadvs2.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( advs3,               os.path.join( folder, 'Example{}-A-CL3-{}.png'.format( examples[ labelBatch[ i ] ][ 'total' ], labelStr[ cadvs3.max( 1 )[ 1 ] ] ) ) )
               utils.save_image( advs1 - orig1,       os.path.join( folder,    'Example{}-D-CL1.png'.format( examples[ labelBatch[ i ] ][ 'total' ] ) ) )
               utils.save_image( advs2 - orig2,       os.path.join( folder,    'Example{}-D-CL2.png'.format( examples[ labelBatch[ i ] ][ 'total' ] ) ) )
               utils.save_image( advs3 - orig3,       os.path.join( folder,    'Example{}-D-CL3.png'.format( examples[ labelBatch[ i ] ][ 'total' ] ) ) )
               
               filename = os.path.join( folder, 'classification.csv' )
               if( not os.path.exists( filename ) ):
                  with open( filename, 'w', newline='' ) as csvfile:
                     csvWriter = csv.writer( csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL )
                     csvWriter.writerow( [ 'OL1', 'OL2', 'OL3', 'AL1', 'AL2', 'AL3' ] )
      
               with open( filename, 'a+', newline='' ) as csvfile:
                  csvWriter = csv.writer( csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL )
                  csvWriter.writerow( [ labelStr[ corig1.max( 1, keepdim = True )[ 1 ] ], labelStr[ corig2.max( 1, keepdim = True )[ 1 ] ], labelStr[ corig3.max( 1, keepdim = True )[ 1 ] ], 
                                        labelStr[ cadvs1.max( 1, keepdim = True )[ 1 ] ], labelStr[ cadvs2.max( 1, keepdim = True )[ 1 ] ], labelStr[ cadvs3.max( 1, keepdim = True )[ 1 ] ] ] )
         # Update Progress
         progress.Update( batchIndex, len( results ), '' )
      with open( "Images/Epsilon{}/Statistics.csv".format( args.epsilon ), 'w', newline='' ) as csvfile:
         csvWriter = csv.writer( csvfile, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL )
         csvWriter.writerow( [ 'label', 'Total Examples', 'CL1 Mismatch', 'CL2 Mismatch', 'CL3 Mismatch' ] )
         for i in range( len( examples ) ):
            csvWriter.writerow( [ labelStr[ i ], examples[ i ][ 'total' ], examples[ i ][ 'cl1' ], examples[ i ][ 'cl2' ], examples[ i ][ 'cl3' ] ] )

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
   
