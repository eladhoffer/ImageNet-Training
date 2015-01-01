require 'cunn'
require 'cudnn'
local model = nn.Sequential() 

-- Convolution Layers

model:add(cudnn.SpatialConvolution(3, 96, 11,11,4,4))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(96, 96,1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(96,96, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3, 3,2,2):ceil())
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(96,256, 5,5,1,1,2,2 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256,256, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256,256, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3, 3,2,2):ceil())
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(256,384, 3,3 ,1,1,1,1))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(384,384, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(384,384, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(3, 3,2,2):ceil())
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(384,1024, 3,3,1,1,1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(1024,1024, 1,1 ))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(1024,1000, 1,1 ))
model:add(cudnn.ReLU())

model:add(cudnn.SpatialAveragePooling(6,6))
model:add(nn.View(1000))
model:add(nn.LogSoftMax())

return model

