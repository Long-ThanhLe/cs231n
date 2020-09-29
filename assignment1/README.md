Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2019.

# Note

## 1. Image classification

### Challenges:

- Viewpoint variation: A single instance of an object can be oriented in many ways with respect to the camera.
- Scale variation
- Deformation: Many objects of interest are not rigid bodies and can be
  deformed in extreme ways
- Occlusion: The objects of interest can be occluded. Sometimes only a
  small portion of an object (as little as few pixels) could be visible
- Illumination conditions: The effects of illumination are drastic on the pixel level.
- Background clutter: The objects of interest may blend into their environment, making them hard to identify.
- Intra-class variation: The classes of interest can often be relatively broad, such as chairs. There are many different types of these objects, each with their own appearance

### Data-driven approach:

- Relies on first accumulating a *training dataset* of labeled images.

### CIFAR-10

- 60,000 images with size 32x32, 10 classes.
- Training set: 50,000
- Test set: 10,000

### Nearest Neighbor Classifier
- Take a test image, compare it to every single one of the training images,
  and predict the label of the closest training image.
- The choices of distance: $$L1$$, $$L2$$, ...

### k-Nearest Neighbor Classifier
- Instead of finding the single closest image in the training set, we will find the top **k** closest images and have them vote on the label of the test image.
### Validation sets for Hyperparameter tuning
- Split training set
- Cross-validation: Instead of arbitrarily picking the first 1000 data points to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of $$k$$ works by iterating over different validation sets and averaging the performance across these.
### Pros and Cons of Nearest Neighbor classifier
#### Advantage
- Simple to implement and understand
- Classifier takes no time to train
#### Disadvantage
- Computational cost at test time
- Distances over high-dimensional spaces can be very counter-intuitive
## 2. Linear Classification

### Linear mapping

- $$f(x_i,W,b)=Wx_i+b$$
- The single matrix multiplication $$W_xi$$ is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of $$W$$.
### Interpreting a linear classifier

**As high-dimensional points**

- Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space
- Since we defined the score of each class as a weighted sum of all image pixels, each class score is a linear function over this space

**As template matching**

- Each row of $$W$$ corresponds to a *template* (or sometimes also called a *prototype*) for one of the classes
- The score of each class for an image is then obtained by comparing each template with the image using an *inner product* (or *dot product*) one by one to find the one that “fits” best.
**Bias trick**
- $$f(x_i,W,b)=Wx_i+b$$ $$->$$ $$f(x_i,W)=Wx_i$$

### Loss function

#### mSVM loss

- The score function takes the pixels and computes the vector $$f(x_i,W)$$ of class scores, which we will abbreviate to $$s$$ (short for scores).
- $$Li=\sum_{j≠yi} max( 0, s_{j}−s_{yi}+\Delta )$$
- In linear classifier: $$Li=\sum_{j≠yi} max( 0, w^T_{j}x_i−w^T_{y_i}x_i+ \Delta )$$,  $$w_i$$ is the i_th row of $$W$$ reshaped as a column
- The theshold at zero $$max(0,-)$$ function is called **hingle loss**
- **Squared hingle loss SVM** (L2-SVM): $$max(0,-)^2$$
- Regularization:
  - There might be many similar **W** that correctly classify the examples
  - if some parameters **W** correctly classify all examples (so loss is zero for each example), then any multiple of these parameters $$λW$$ where $$λ>1$$ will also give zero loss because this transformation uniformly stretches all score magnitudes and hence also their absolute differences
  - -> Extending the loss function with a **regularization penalty** $$R(W)$$
  - $$L2$$ penalty: $$R(W) = \sum_k\sum_lW^2_{k,l}$$
  - Loss becomes:
    $$L = \frac{1}{N}\sum_iL_i + \lambda R(W)$$
  - Max margin property in SVM: **CS229**
  - Generalization property -> less overfitting

### Practical Considerations

- Setting $$\Delta$$:
  - This hyperparameter can safely be set to $$Δ=1.0$$ in all cases.
  - The hyperparameters $$\Delta$$ and $$λ$$ seem like two different hyperparameters, but in fact, they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective
  - The exact value of the margin between the scores (e.g. $$Δ=1$$, or $$Δ=100$$) is in some sense meaningless because the weights can shrink or stretch the differences arbitrarily.

### Softmax classifier

- Cross-entropy loss:
  $$
  L_i = -log(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}})  = -f_{y_i} + log\sum_j e^{f_j}
  $$
- $$f_j$$ is the j-th element of the vector of class scores $$f$$

#### Information theory view:

- The cross-entropy between a 'true' distribution $$p$$ and an estimated distribution $$q$$ is defined as:
  $$
  H(p,q) = -\sum_x p(x) log q(x)
  $$
- The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ($$q=\frac{e^{f_{yi}}}{∑_j e^{f_j}}$$ as seen above) and the “true” distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e.$$p=[0,…1,…,0]$$ contains a single $$ 1$$ at the $$y_i$$ i-th position.)
- Since the cross-entropy can be written in terms of entropy and the Kullback-Leibler divergence as $$H(p,q)=H(p)+DKL(p||q)$$, and the entropy of the delta function pp is zero, this is also equivalent to minimizing the KL divergence between the two distributions.

#### Probabilistic Interpretation

- We can see 

  $$
  \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
  $$


  as the normalized probability assigned to the correct label $$y_i$$ given the image $$x_i$$
- In the probabilistic interpretation, we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing *Maximum Likelihood Estimation* (MLE)
- A nice feature of this view is that we can now also interpret the regularization term $$R(W)$$ in the full loss function as coming from a Gaussian prior over the weight matrix $$W$$, where instead of MLE we are performing the *Maximum a posteriori* (MAP) estimation

### SVM vs Softmax

#### Softmax classifier provides probabilities for each class

#### SVM and Softmax are usually comparable

- Compared to the Softmax classifier, the SVM is a more *local* objective
  - The SVM does not care about the details of the individual scores: if they were instead [10, -100, -100] or [10, 9, 9] the SVM would be indifferent since the margin of 1 is satisfied and hence the loss is zero
- The Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better

## 3. Optimization

### Optimization

#### Random search

- Try random weights and keep track of what works best

#### Random local search

- We will start out with a random $W$, generate random perturbations δWδW to it and if the loss at the perturbed $W+δW$ is lower, we will perform an update

#### Following the gradient

- It turns out that there is no need to randomly search for a good direction: we can compute the *best* direction along which we should change our weight vector that is mathematically guaranteed to be the direction of the steepest descend

### Computing the gradient

#### Computing the gradient numerically with finite differences

- Following the gradient formula we gave above, the code above iterates overall dimensions one by one, makes a small change `h` along that dimension, and calculates the partial derivative of the loss function along that dimension by seeing how much the function changed. The variable `grad` holds the full gradient in the end
- it often works better to compute the numeric gradient using the **centered difference formula**
- the numerical gradient has complexity linear in the number of parameters

Computing the gradient with Calculus

- ...

### Gradient Descent

#### Mini-batch gradient descent

- Compute the gradient over **batches** of the training data.
- The gradient from a mini-batch is a good approximation of the gradient of the full objective

#### Stochastic Gradient Descent (SGD)

- The extreme case of this is a setting where the mini-batch contains only a single example

## Related question/keyword/notes

- CS229: Max margin property in SVM
- https://stanford.edu/~boyd/cvxbook/
- https://en.wikipedia.org/wiki/Subderivative
-
