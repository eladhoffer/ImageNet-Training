Deep Learning on ImageNet using Torch
=====================================
This is a complete training example for Deep Convolutional Networks on the ILSVRC classification task.

Data is preprocessed and cached as a LMDB data-base for fast reading. A separate thread buffers images from the LMDB record in the background. 

Multiple GPUs are also supported by using nn.DataParallelTable (https://github.com/torch/cunn/blob/master/docs/cunnmodules.md).

This code allows training at 4ms/sample with the AlexNet model and 2ms for testing on a single GPU (using Titan Z with 1 active gpu)

##Dependencies
* "eladtools" (https://github.com/eladhoffer/eladtools) for DataProvider class and optimizer.
* “lmdb.torch” (http://github.com/eladhoffer/lmdb.torch) for LMDB usage.
* “cudnn.torch” (https://github.com/soumith/cudnn.torch) for faster training. Can be avoided by changing "cudnn" to "nn" in models.


##Data
* To get the ILSVRC data, you should register on their site for access: http://www.image-net.org/
* Configure the data location and save dir in **Config.lua**.
* LMDB records for fast read access are created by running **CreateLMDBs.lua**. 
It defaults to saving the compressed jpgs (about ~20GB for training data, ~1GB for validation data).


##Training
You can start training using:
```lua
th Main.lua -network AlexNet -LR 0.01
```
or if you have 2 gpus availiable,
```lua
th Main.lua -network AlexNet -LR 0.01 -nGPU 2 -batchSize 256
```

Buffer size should be adjusted to suit the used hardware and configuration. Default value is 1280 (10 batches of 128) which works well when using a non SSD drive and 1 GPU.

##Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | Models Folder
|network          |  AlexNet             | Model file - must return valid network.
|LR               |  0.01                | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples
|weightDecay      |  5e-4                | L2 penalty on the weights
|momentum         |  0.9                 | momentum
|batchSize        |  128                 | batch size
|optimization     |  sgd                 | optimization method
|epoch            |  -1                  | number of epochs to train (-1 for unbounded)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|bufferSize       |  1280                | buffer size
|devid            |  1                   | device ID (if using CUDA)
|nGPU             |  1                   | num of gpu devices used
|load             |  none                | load existing net weights
|save             |  time-identifier     | save directory
|shuffle          |  true               | shuffle training samples
