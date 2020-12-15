import sys
import argparse
from Model.CNN import CNN
from Utility.Data import Data

def main( ):
   print( "Parsing Arguments" )
   parser = argparse.ArgumentParser( description = "CPSC-597" )
   parser.add_argument( '--dataset',    type = str, default = 'mnist', choices = [ 'mnist' ] )
   parser.add_argument( '--data_path',  type = str, required = True,   help = 'path to dataset' )
   parser.add_argument( '--batch_size', type = int, default = 64,      help = 'Size of a batch' )
   parser.add_argument( '--mode',       type = str, default = 'train', choices = [ 'train', 'test' ] )
   parser.add_argument( '--epochs',     type = int, default = 50,      help = 'The number of epochs to run')
   parser.add_argument( '--cuda',       type = str, default = True,    help = 'Availability of cuda' )
   args = parser.parse_args( )

   print( "Creating Dataloader" )
   data = Data( args.dataset, args.data_path, args.batch_size )

   print( "Creating Model" )
   model = CNN( 1, args.cuda )

   if( args.mode == 'train' ):
      model.Train( data.train_loader, args.epochs, args.batch_size )

if __name__ == "__main__":
   sys.exit( int( main( ) or 0 ) )
   
