## @package AdversarialDetection
# 
import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torchvision import utils
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
      epsilons = [0, .05, .1, .15, .2, .25, .3]
      #grid = utils.make_grid( images[ 0 ][ 2 ] )
      #utils.save_image( grid, 'img_generatori_iter_{}.png'.format( str( 0 ).zfill( 3 ) ) )
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
          acc, ex = TestAttack(model, data.test_loader, eps)
          accuracies.append(acc)
          examples.append(ex)

      ######################################################################
      # Results
      # -------
      # 
      # Accuracy vs Epsilon
      # ~~~~~~~~~~~~~~~~~~~
      # 
      # The first result is the accuracy versus epsilon plot. As alluded to
      # earlier, as epsilon increases we expect the test accuracy to decrease.
      # This is because larger epsilons mean we take a larger step in the
      # direction that will maximize the loss. Notice the trend in the curve is
      # not linear even though the epsilon values are linearly spaced. For
      # example, the accuracy at :math:`\epsilon=0.05` is only about 4% lower
      # than :math:`\epsilon=0`, but the accuracy at :math:`\epsilon=0.2` is 25%
      # lower than :math:`\epsilon=0.15`. Also, notice the accuracy of the model
      # hits random accuracy for a 10-class classifier between
      # :math:`\epsilon=0.25` and :math:`\epsilon=0.3`.
      # 
      plt.figure(figsize=(5,5))
      plt.plot(epsilons, accuracies, "*-")
      plt.yticks(np.arange(0, 1.1, step=0.1))
      plt.xticks(np.arange(0, .35, step=0.05))
      plt.title("Accuracy vs Epsilon")
      plt.xlabel("Epsilon")
      plt.ylabel("Accuracy")
      plt.show()

      ######################################################################
      # Sample Adversarial Examples
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # 
      # Remember the idea of no free lunch? In this case, as epsilon increases
      # the test accuracy decreases **BUT** the perturbations become more easily
      # perceptible. In reality, there is a tradeoff between accuracy
      # degredation and perceptibility that an attacker must consider. Here, we
      # show some examples of successful adversarial examples at each epsilon
      # value. Each row of the plot shows a different epsilon value. The first
      # row is the :math:`\epsilon=0` examples which represent the original
      # “clean” images with no perturbation. The title of each image shows the
      # “original classification -> adversarial classification.” Notice, the
      # perturbations start to become evident at :math:`\epsilon=0.15` and are
      # quite evident at :math:`\epsilon=0.3`. However, in all cases humans are
      # still capable of identifying the correct class despite the added noise.
      # 
      # Plot several examples of adversarial samples at each epsilon
      cnt = 0
      plt.figure(figsize=(8,10))
      for i in range(len(epsilons)):
          for j in range(len(examples[i])):
              cnt += 1
              plt.subplot(len(epsilons),len(examples[0]),cnt)
              plt.xticks([], [])
              plt.yticks([], [])
              if j == 0:
                  plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
              orig,adv,ex = examples[i][j]
              plt.title("{} -> {}".format(orig, adv))
              plt.imshow(ex, cmap="gray")
      plt.tight_layout()
      plt.show()

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

def FGSMAttack( image, epsilon, data_grad ):
   # Collect the element-wise sign of the data gradient
   sign_data_grad = data_grad.sign()
   # Create the perturbed image by adjusting each pixel of the input image
   perturbed_image = image + epsilon*sign_data_grad
   # Adding clipping to maintain [0,1] range
   perturbed_image = torch.clamp(perturbed_image, 0, 1)
   # Return the perturbed image
   return perturbed_image

def TestAttack( model, test_loader, epsilon ):
   # Accuracy counter
   correct = 0
   adv_examples = []
   device = "cuda"

   # Loop over all examples in test set
   for data, target in test_loader:

      # Send the data and label to the device
      data, target = data.to(device), target.to(device)

      # Set requires_grad attribute of tensor. Important for Attack
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)
      init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

      # If the initial prediction is wrong, dont bother attacking, just move on
      if init_pred[ 0 ].item() != target[ 0 ].item():
         continue

      # Calculate the loss
      loss = F.nll_loss(output, target)

      # Zero all existing gradients
      model.zero_grad()

      # Calculate gradients of model in backward pass
      loss.backward()

      # Collect datagrad
      data_grad = data.grad.data

      # Call FGSM Attack
      perturbed_data = FGSMAttack(data, epsilon, data_grad)

      # Re-classify the perturbed image
      output = model(perturbed_data)

      # Check for success
      final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      if final_pred[ 0 ].item() == target[ 0 ].item():
         correct += 1
         # Special case for saving 0 epsilon examples
         if (epsilon == 0) and (len(adv_examples) < 5):
               adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
               adv_examples.append( (init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex) )
      else:
         # Save some adv examples for visualization later
         if len(adv_examples) < 5:
               adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
               adv_examples.append( (init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex) )

   # Calculate final accuracy for this epsilon
   final_acc = correct/float(len(test_loader))
   print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

   # Return the accuracy and an adversarial example
   return final_acc, adv_examples

if __name__ == "__main__":
   sys.exit( int( Main( ) or 0 ) )
   
