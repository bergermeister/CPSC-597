import torch
import foolbox
import Utility.DataManagerPytorch as DMP
from torch.nn import functional as F
from Utility.ProgressBar import ProgressBar

class Adversary( object ):
   def __init__( self, model ):
      self.model = model
      self.activation = {}

   def CreateImage( self, image, epsilon, data_grad ):
      # Collect the element-wise sign of the data gradient
      sign_data_grad = data_grad.sign()
      # Create the perturbed image by adjusting each pixel of the input image
      perturbed_image = image + ( epsilon * sign_data_grad )
      # Adding clipping to maintain [0,1] range
      perturbed_image = torch.clamp( perturbed_image, 0, 1 )
      # Return the perturbed image
      return( perturbed_image )

   def Attack( self, test_loader, epsilon ):
      def getActivation( name ):
         def hook( model, input, output ):
            self.activation[ name ] = output.detach( )
         return( hook )

      # Attach hook for activations
      self.model.cl1.register_forward_hook( getActivation( 'cl1' ) )
      self.model.cl2.register_forward_hook( getActivation( 'cl2' ) )
      self.model.cl3.register_forward_hook( getActivation( 'cl3' ) )

      # Accuracy counter
      correct = 0
      adv_examples = []
      device = "cuda"

      # Loop over all examples in test set
      for data, target in test_loader:

         # Send the data and label to the device
         data, target = data.to( device ), target.to( device )

         # Set requires_grad attribute of tensor. Important for Attack
         data.requires_grad = True

         # Forward pass the data through the model
         self.activation = {}
         output = self.model( data )
         init_pred = output.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
         init_act  = self.activation
         init_int  = self.model.interm

         # If the initial prediction is wrong, dont bother attacking, just move on
         if init_pred[ 0 ].item( ) != target[ 0 ].item( ):
            continue

         # Calculate the loss
         loss = F.nll_loss( output, target )

         # Zero all existing gradients
         self.model.zero_grad( )

         # Calculate gradients of model in backward pass
         loss.backward( )

         # Collect datagrad
         data_grad = data.grad.data

         # Call FGSM Attack
         perturbed_data = self.CreateImage( data, epsilon, data_grad )

         # Re-classify the perturbed image
         self.activation = {}
         output = self.model( perturbed_data )
         final_act = self.activation
         final_int = self.model.interm

         # Check for success
         final_pred = output.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
         if final_pred[ 0 ].item( ) == target[ 0 ].item( ):
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                  adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                  adv_examples.append( ( init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex, init_act, final_act, init_int, final_int ) )
         else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                  adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                  adv_examples.append( ( init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex, init_act, final_act, init_int, final_int ) )

      # Calculate final accuracy for this epsilon
      final_acc = correct/float(len(test_loader))
      print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

      # Return the accuracy and an adversarial example
      return( final_acc, adv_examples )

   #PGD attack method using Foolbox
   #Returns a dataloader with the adversarial samples and the target labels  
   def PGDAttack( self, device, dataLoader, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted ):
      self.model.eval() #Change model to evaluation mode for the attack 
      #Wrap the model using Foolbox's Pytorch wrapper 
      fmodel = foolbox.PyTorchModel( self.model, bounds=(clipMin, clipMax))
      #Create attack variable 
      attack = foolbox.attacks.LinfPGD(abs_stepsize=epsilonStep, steps=numSteps)
      #Generate variables for storing the adversarial examples 
      numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
      xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
      xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
      yClean = torch.zeros(numSamples, dtype = torch.long)
      advSampleIndex = 0 
      batchSize = 0 #just do dummy initalization, will be filled in later 
      tracker = 0

      result = []
      #Go through and generate the adversarial samples in batches 
      for i, (xCurrent, yCurrent) in enumerate(dataLoader):
         batchSize = xCurrent.shape[0] #Get the batch size so we know indexing for saving later
         tracker = tracker + batchSize
         #print("Processing up to sample=", tracker)
         xCurrentCuda = xCurrent.to(device) #Load the data into the GPU
         yCurrentCuda = yCurrent.type(torch.LongTensor).to(device)
         if targeted == True:
            criterion = foolbox.criteria.TargetedMisclassification(yCurrentCuda)
         else:
            criterion = foolbox.criteria.Misclassification(yCurrentCuda)
         #Next line actually runs the attack 
         _, advs, success = attack(fmodel, xCurrentCuda, epsilons=epsilonMax, criterion=criterion)
         result.append( { 'status' : success, 'orig' : xCurrent, 'adv': advs, 'label' : yCurrent } )
         #Save the adversarial samples 
         for j in range(0, batchSize):
            xAdv[advSampleIndex] = advs[j]
            yClean[advSampleIndex] = yCurrent[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
      #All samples processed, now time to save in a dataloader and return 
      advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
      return advLoader, result

   def CreateSuccessLoader( self, device, dataLoader ):
      self.model.eval( ) #change Model to evaluation mode
      successData = []

      progress = ProgressBar( 40, 80 )
      testLoss = 0
      total    = 0
      correct  = 0
      print( 'Begin Evaluation...' )
      with torch.no_grad( ):
         for batchIndex, ( images, labels ) in enumerate( dataLoader ):
            # Forward Pass
            images = images.to( device )
            labels = labels.to( device )
            outputs = self.model( images )
            _, predicted = outputs.max( 1 )
            total    += labels.size( 0 )
            correct  += predicted.eq( labels ).sum( ).item( )

            # Capture successful data
            for i in range( 0, images.shape[ 0 ] ):
               if( predicted[ i ].eq( labels[ i ] ) ):
                  successData.append( { 'image': images[ i ], 'label': labels[ i ] } )

            # Update Progress
            progress.Update( batchIndex, len( dataLoader ), ' Acc: %.3f%% (%d/%d)'
                             % ( 100. * correct / total, correct, total ) )

      print( 'End Evaluation...' )
      print( 'Creating Success Loader...' )
      xShape   = DMP.GetOutputShape( dataLoader )
      xSuccess = torch.zeros( len( successData ), xShape[ 0 ], xShape[ 1 ], xShape[ 2 ] )
      ySuccess = torch.zeros( len( successData ), dtype = torch.long )
      for i in range( len( successData ) ):
         xSuccess[ i ] = successData[ i ][ 'image' ]
         ySuccess[ i ] = successData[ i ][ 'label' ]
      successLoader = DMP.TensorToDataLoader( xSuccess, ySuccess, transforms = None, batchSize = dataLoader.batch_size, randomizer = None ) #use the same batch size as the original loader
      print( 'Success Loader Created...' )
      return( successLoader )
   
   def CreateDetectorLoader( self, cnn, recon, dataLoader, advLoader ):
      print( 'Creating Detector Loader...' )
      metaData = []

      cnn.eval()
      recon.eval()

      for batchIndex, ( images, labels ) in enumerate( dataLoader ):
         if( cnn.cudaEnable ):
            images, labels = images.cuda( ), labels.cuda( )
         with torch.no_grad():
            out1 = images
            out2, out3, out4 = recon( { 'model' : cnn, 'input' : images } )
            out = torch.cat( ( out1, out2, out3, out4 ), 1 )
         for i in range( 0, out.shape[ 0 ] ):
            metaData.append( { 'image': out[ i ], 'label': 0 } )

      for batchIndex, ( images, labels ) in enumerate( advLoader ):
         if( cnn.cudaEnable ):
            images, labels = images.cuda( ), labels.cuda( )
         with torch.no_grad():
            out1 = images
            out2, out3, out4 = recon( { 'model' : cnn, 'input' : images } )
            out = torch.cat( ( out1, out2, out3, out4 ), 1 )
         for i in range( 0, out.shape[ 0 ] ):
            metaData.append( { 'image': out[ i ], 'label': 1 } )

      xSuccess = torch.zeros( len( metaData ), metaData[ 0 ][ 'image' ].size( )[ 0 ], metaData[ 0 ][ 'image' ].size( )[ 1 ], metaData[ 0 ][ 'image' ].size( )[ 2 ] )
      ySuccess = torch.zeros( len( metaData ), dtype = torch.long )
      for i in range( len( metaData ) ):
         xSuccess[ i ] = metaData[ i ][ 'image' ]
         ySuccess[ i ] = metaData[ i ][ 'label' ]
      metaData = DMP.TensorToDataLoader( xSuccess, ySuccess, transforms = None, batchSize = dataLoader.batch_size, randomizer = True ) #use the same batch size as the original loader
      print( 'Detector Loader Created...' )
      return( metaData )

   def CreateMetaLoader( self, dataLoader, advLoader ):
      print( 'Creating Meta CNN Loader...' )
      metaData = []

      for batchIndex, ( images, labels ) in enumerate( dataLoader ):
         images, labels = images.cuda( ), labels.cuda( )
         for i in range( 0, images.shape[ 0 ] ):
            metaData.append( { 'image': images[ i ], 'label': labels[ i ] } )

      for batchIndex, ( images, labels ) in enumerate( advLoader ):
         images, labels = images.cuda( ), labels.cuda( )
         for i in range( 0, images.shape[ 0 ] ):
            metaData.append( { 'image': images[ i ], 'label': 10 } )

      xSuccess = torch.zeros( len( metaData ), metaData[ 0 ][ 'image' ].size( )[ 0 ], metaData[ 0 ][ 'image' ].size( )[ 1 ], metaData[ 0 ][ 'image' ].size( )[ 2 ] )
      ySuccess = torch.zeros( len( metaData ), dtype = torch.long )
      for i in range( len( metaData ) ):
         xSuccess[ i ] = metaData[ i ][ 'image' ]
         ySuccess[ i ] = metaData[ i ][ 'label' ]
      metaData = DMP.TensorToDataLoader( xSuccess, ySuccess, transforms = None, batchSize = dataLoader.batch_size, randomizer = True ) #use the same batch size as the original loader
      print( 'Meta CNN Loader Created...' )
      return( metaData )
   