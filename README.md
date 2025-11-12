We are going to use Convulutional NN that work on any kind of grid-like  or sequential data, including time series data from accelerometers signals. 

We will use Conv1D which are fast, low memeory, parralleizable, and have good accuracy. 

We will quantize our model to uint8 so that we will have 1 byte per model weight.

We have 10 classes that we are predicting.  