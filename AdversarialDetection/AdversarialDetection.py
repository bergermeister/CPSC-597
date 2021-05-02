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
from Model.Reconstruct import Reconstruct
from Model.Encoder import Encoder
from Model.Classifier import Classifier
from Model.Adversary import Adversary
from Model.MetaClassifier import MetaClassifier
from Model.MetaCNN import MetaCNN

from Utility.Data import Data
from Utility.ProgressBar import ProgressBar

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
   parser.add_argument( '--mode',       type = str, default = 'train',           choices = [ 'encoder', 'classifier', 'train', 'atrain', 'metatrain', 'metacnn', 'test', 'recon', 'adversary' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,                help = 'The number of epochs to run')
   parser.add_argument( '--epsilon',    type = float, default = 0.1,             help = 'Perturbed image pixel adjustment factor' )
   parser.add_argument( '--cuda',       type = str, default = 'True',            help = 'Availability of cuda' )
   parser.add_argument( '--cnn',        type = str, default = 'cnn.state',       help = 'Path to CNN Model State' )
   parser.add_argument( '--ann',        type = str, default = 'ann.state',       help = 'Path to ACNN Model State' )
   parser.add_argument( '--recon',      type = str, default = 'recon.state',     help = 'Path to Reconstruction Model State' )
   parser.add_argument( '--encoder',    type = str, default = 'encoder.state',   help = 'Path to Encoder Model State' )
   parser.add_argument( '--classifier', type = str, default = 'clasifier.state', help = 'Path to Classifier Model State' )
   parser.add_argument( '--metann',     type = str, default = 'metann.state',    help = 'Path to Meta Classifier Model State' )
   parser.add_argument( '--metacnn',    type = str, default = 'metacnn.state',   help = 'Path to Meta CNN Model State' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Models" )
   cnn        = CNN( data.channels, args.cuda )
   acnn       = CNN( data.channels, args.cuda )
   recon      = Reconstruct( data.channels, args.cuda )
   encoder    = Encoder( data.channels, args.cuda )
   classifier = Classifier( args.cuda )
   meta       = MetaClassifier( args.cuda )
   metacnn    = MetaCNN( data.channels * 4, args.cuda )

   if( os.path.exists( args.cnn ) ):
      print( "Loading CNN state" )
      cnn.Load( args.cnn )
   if( os.path.exists( args.ann ) ):
      print( "Loading ACNN state" )
      acnn.Load( args.acnn )
   if( os.path.exists( args.recon ) ):
      print( "Loading Reconstructor state" )
      recon.Load( args.recon )
   if( os.path.exists( args.encoder ) ):
      print( "Loading encoder state" )
      encoder.Load( args.encoder )
   if( os.path.exists( args.classifier ) ):
      print( "Loading Classifier state" )
      classifier.Load( args.classifier )
   if( os.path.exists( args.metann ) ):
      print( "Loading Meta Classifier state" )
      meta.Load( args.metann )
   if( os.path.exists( args.metacnn ) ):
      print( "Loading Meta CNN state" )
      metacnn.Load( args.metacnn )

   adversary  = Adversary( cnn )

   if( args.mode == 'train' ):
      cnn.Train( data.train_loader, args.epochs, args.batch_size )
      acc = cnn.Test( data.test_loader, args.batch_size )
      if( acc > cnn.accuracy ):
         Save( cnn, 0, cnn.epochs + args.epochs, args.cnn )
   elif( args.mode == 'recon' ):
      recon.Train( cnn, data.train_loader, args.epochs, args.batch_size )
      Save( recon, 0, recon.epochs + recon.epochs, args.recon )
   elif( args.mode == 'classifier' ):
      classifier.Train( cnn, data.train_loader, args.epochs, args.batch_size )
      acc = classifier.Test( cnn, data.test_loader, args.batch_size )
      if( acc > classifier.accuracy ):
         Save( classifier, acc, classifier.epochs + args.epochs, args.classifier )
   elif( args.mode == 'atrain' ):
      epsilon = args.epsilon
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.train_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      combinedLoader = adversary.CreateCombinedLoader( data.train_loader, advloader )
      acnn.Train( combinedLoader, args.epochs, args.batch_size )
      acc = acnn.Test( data.test_loader, args.batch_size )
      if( acc > acnn.accuracy ):
         Save( acnn, acc, acnn.epochs + args.epochs, args.acnn )
   elif( args.mode == 'metatrain' ):
      epsilon = args.epsilon
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.train_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateMetaLoader( cnn, classifier, successLoader, advloader )
      meta.Train( metaloader, args.epochs, args.batch_size )

      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateMetaLoader( cnn, classifier, data.test_loader, advloader )
      acc = meta.Test( metaloader, args.batch_size )
      if( acc > meta.accuracy ):
         Save( meta, acc, meta.epochs + args.epochs, args.metann )
   elif( args.mode == 'metacnn' ):
      epsilon = args.epsilon
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.train_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateMetaCNNLoader( cnn, recon, successLoader, advloader )
      metacnn.Train( metaloader, args.epochs, args.batch_size )

      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      metaloader = adversary.CreateMetaCNNLoader( cnn, recon, data.test_loader, advloader )
      acc = metacnn.Test( metaloader, args.batch_size )
      if( acc > metacnn.accuracy ):
         Save( metacnn, acc, metacnn.epochs + args.epochs, args.metacnn )
   elif( args.mode == 'test' ):
      cnn.eval()
      acc = cnn.Test( data.test_loader, args.batch_size )

   elif( args.mode == 'encoder' ):
      encoder.Train( data.train_loader, args.epochs, args.batch_size )
      loss = encoder.Test( data.test_loader, args.batch_size )
      Save( encoder, loss, encoder.epochs + args.epochs, args.encoder )


   elif( args.mode == 'adversary' ):
      labelStr = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
      cnn.eval( )
      recon.eval( )
      accuracies = []

      epsilon = args.epsilon
      adversary = Adversary( cnn )
      
      successLoader = adversary.CreateSuccessLoader( 'cuda', data.test_loader )
      accuracies.append( cnn.Test( successLoader, args.batch_size ) )
      print( "Generating Adversarial Images" )
      advloader, results = adversary.PGDAttack( "cuda", successLoader, epsilon, epsilon / 10.0, 10, -1, 1, False )
      print( "Evaluating Adversarial Images" )
      accuracies.append( cnn.Test( advloader, args.batch_size ) )
      #print( "Evaluating Blurred Adversarial Images" )
      #accuracies.append( cnn.TestGaussian( advloader, args.batch_size ) )
      #print( "Evaluating Guassian Highpass Adversarial Images" )
      #accuracies.append( cnn.TestGaussianHighpass( advloader, args.batch_size ) )

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
               folder = 'Images/Epsilon{}/'.format( epsilon )
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
      with open( "Images/Epsilon{}/Statistics.csv".format( epsilon ), 'w', newline='' ) as csvfile:
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
   
