import sys
from time import time

class ProgressBar( object ):
   def __init__( self, barLength, termWidth ):
      self.barLength = barLength
      self.termWidth = termWidth
      self.timeLast = time( )
      self.timeBegin = self.timeLast

   def Update( self, current, total, msg = None ):
      if( current == 0 ):
         self.timeBegin = time( ) # Reset for new bar

      curLen  = int( self.barLength * current / total )
      restLen = int( self.barLength - curLen ) - 1

      sys.stdout.write( ' [' )
      for i in range( curLen ):
         sys.stdout.write( '=' )
      sys.stdout.write( '>' )
      for i in range( restLen ):
         sys.stdout.write( '.' )
      sys.stdout.write( ']' )

      curTime = time( )
      stepTime = curTime - self.timeLast
      self.timeLast = curTime
      timeTotal = curTime - self.timeBegin

      L = [ ]
      #L.append( '  Step: %s' % ProgressBar.FormatTime( stepTime ) )
      L.append( ' Time: %s' % ProgressBar.FormatTime( timeTotal ) )
      if msg:
         L.append( ' | ' + msg )

      msg = ''.join( L )
      sys.stdout.write( msg )
      for i in range( self.termWidth - int( self.barLength )- len( msg ) - 3 ):
         sys.stdout.write( ' ' )

      # Go back to the center of the bar.
      for i in range( self.termWidth - int( self.barLength / 2 ) + 2 ):
         sys.stdout.write( '\b' )
      #sys.stdout.write( ' %d/%d ' % ( current + 1, total ) )

      if( current < ( total - 1 ) ):
         sys.stdout.write( '\r' )
      else:
         sys.stdout.write( '\n' )
      sys.stdout.flush( )

   def FormatTime( seconds ):
      millis = int( seconds * 1000 )
      return( str( millis ) + 'ms' )
