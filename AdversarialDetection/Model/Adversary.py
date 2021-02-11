import torch
from torch.nn import functional as F

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
         init_act  = self.activation[ 'cl1' ].squeeze( )

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
         final_act = self.activation[ 'cl1' ].squeeze( )

         # Check for success
         final_pred = output.max( 1, keepdim = True )[ 1 ] # get the index of the max log-probability
         if final_pred[ 0 ].item( ) == target[ 0 ].item( ):
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                  adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                  adv_examples.append( ( init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex, init_act, final_act ) )
         else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                  adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                  adv_examples.append( ( init_pred[ 0 ].item( ), final_pred[ 0 ].item( ), adv_ex, init_act, final_act ) )

      # Calculate final accuracy for this epsilon
      final_acc = correct/float(len(test_loader))
      print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

      # Return the accuracy and an adversarial example
      return( final_acc, adv_examples )

   #def Plot( self ):
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
      #plt.figure(figsize=(5,5))
      #plt.plot(epsilons, accuracies, "*-")
      #plt.yticks(np.arange(0, 1.1, step=0.1))
      #plt.xticks(np.arange(0, .35, step=0.05))
      #plt.title("Accuracy vs Epsilon")
      #plt.xlabel("Epsilon")
      #plt.ylabel("Accuracy")
      #plt.show()

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
      #cnt = 0
      #plt.figure(figsize=(8,10))
      #for i in range(len(epsilons)):
      #    for j in range(len(examples[i])):
      #        cnt += 1
      #        plt.subplot(len(epsilons),len(examples[0]),cnt)
      #        plt.xticks([], [])
      #        plt.yticks([], [])
      #        if j == 0:
      #            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
      #        orig,adv,ex = examples[i][j]
      #        plt.title("{} -> {}".format(orig, adv))
      #        plt.imshow(ex, cmap="gray")
      #plt.tight_layout()
      #plt.show()
