# 1. Image classification

### Challenges:

- Viewpoint variation: A single instance of an object can be oriented in
  many ways with respect to the camera.
- Scale variation
- Deformation: Many objects of interest are not rigid bodies and can be
  deformed in extreme ways
- Occlusion: The objects of interest can be occluded. Sometimes only a
  small portion of an object (as little as few pixels) could be visible
- Illumination conditions: The effects of illumination are drastic on the
  pixel level.
- Background clutter: The objects of interest may blend into their
  environment, making them hard to identify.
- Intra-class variation: The classes of interest can often be relatively
  broad, such as chair. There are many different types of these objects,
  each with their own appearance

### Data-driven approach:

- Relies on first accumulating a *training dataset* of labeled images.

### CIFAR-10

- 60,000 images with size 32x32, 10 classes.
- Training set: 50,000
- Test set: 10,000

### Nearest Neighbor Classifier

- Take a test image, compare it to every single one of the training images,
  and predict the label of the closest training image.
- The choices of distance: L1, L2, ...

### k-Nearest Neighbor Classifier

- Instead of finding the single closest image in the training set, we will find the top **k** closest images, and have them vote on the label of the test image.

### Validation sets for Hyperparameter tuning

- Split training set
- Cross-validation: Instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of *k* works by iterating over different validation sets and averaging the performance across these.

### Pros and Cons of Nearest Neighbor classifier

Advantage:

- Simple to implement and understand
- Classifier takes no time to train

Disadvantage:

- Computational cost at test time
- Distances over high-dimensional spaces can be very counter-intuitive

# 2. Linear Classification
