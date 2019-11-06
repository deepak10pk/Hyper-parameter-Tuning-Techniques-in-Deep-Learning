# Hyper-parameter-Tuning-Techniques-in-Deep-Learning
# 1.Gradient Descent
  * A gradient is basically the slope of a function; the degree of change of a parameter with the amount of change in another parameter. Mathematically, it can be described as the partial derivatives of a set of parameters with respect to its inputs. The more the gradient, the steeper the slope.
  * Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.
  * Starting at the top of the mountain, we take our first step downhill in the direction specified by the negative gradient. Next we recalculate the negative gradient (passing in the coordinates of our new point) and take another step in the direction it specifies. We continue this process iteratively until we get to the bottom of our graph, or to a point where we can no longer move downhill–a local minimum ---Please refer to following figure:
      ![title](gd.png)
 
 # Stochastic Gradient Descent (SGD)
   * The word ‘stochastic‘ means a system or a process that is linked with a random probability. Hence, in Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration.
   * In Gradient Descent, there is a term called “batch” which denotes the total number of samples from a dataset that is used for calculating the gradient for each iteration.
   *  In typical Gradient Descent optimization, like Batch Gradient Descent, the batch is taken to be the whole dataset. Although, using the whole dataset is really useful for getting to the minima in a less noisy or less random manner, but the problem arises when our datasets get really huge.
   * Suppose, you have a million samples in your dataset, so if you use a typical Gradient Descent optimization technique, you will have to use all of the one million samples for completing one iteration while performing the Gradient Descent, and it has to be done for every iteration until the minima is reached. Hence, it becomes computationally very expensive to perform.
   * In SGD, since only one sample from the dataset is chosen at random for each iteration, the path taken by the algorithm to reach the minima is usually noisier than your typical Gradient Descent algorithm. But that doesn’t matter all that much because the path taken by the algorithm does not matter, as long as we reach the minima and with significantly shorter training time.
   # Path taken by Batch Gradient Descent – ->>  ![title](gdp.png)
   # Path taken by Stochastic Gradient Descent –> ![title](sgd-1.png)

# Gradient Descent with Momentum
  * Mini-batch gradient descent makes a parameter update with just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will “oscillate” toward convergence. Gradient Descent with Momentum considers the past gradients to smooth out the update. It computes an exponentially weighted average of your gradients, and then use that gradient to update your weights instead. It works faster than the standard gradient descent algorithm.
  
   # How does it work? 
   Consider an example where we are trying to optimize a cost function which has contours like below and the red dot denotes the position of the local optima (minimum).![title](Fig1-2-768x151.png)
  
  We start gradient descent from point ‘A’ and after one iteration of gradient descent we may end up at point ‘B’, the other side of the ellipse. Then another step of gradient descent may end up at point ‘C’. With each iteration of gradient descent, we move towards the local optima with up and down oscillations. If we use larger learning rate then the vertical oscillation will have higher magnitude. So, this vertical oscillation slows down our gradient descent and prevents us from using a much larger learning rate.
  ![title](Fig2-2.png)
  
 By using the exponentially weighted average values of dW and db, we tend to average out the oscillations in the vertical direction closer to zero as they are in both directions (positive and negative). Whereas, on the horizontal direction, all the derivatives are pointing to the right of the horizontal direction, so the average in the horizontal direction will still be pretty big. It allows our algorithm to take more straight forwards path towards local optima and damp out vertical oscillations. Due to this reason, the algorithm will end up at local optima with a few iterations.   ![title](Fig3-2-768x167.png) 
 
#  How to Implement?

During backward propagation, we use dW and db to update our parameters W and b as follows:

W = W – learning rate * dW

b = b – learning rate * db

In momentum, instead of using dW and db independently for each epoch, we take the exponentially weighted averages of dW and db.

VdW = β x VdW + (1 – β) x dW

Vdb = β x Vdb + (1 – β) x db

Where beta ‘β’ is another hyperparameter called momentum and ranges from 0 to 1. It sets the weight between the average of previous values and the current value to calculate the new weighted average.

After calculating exponentially weighted averages, we will update our parameters.

W = W – learning rate *VdW

b = b – learning rate * Vdb


   
      
     


   
           
    
  
