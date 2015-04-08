

require 'cudnn'
require 'cunn'
local SpatialConvolution = cudnn.SpatialConvolution
local SpatialMaxPooling = cudnn.SpatialMaxPooling
local ReLU = cudnn.ReLU

local features = nn.Sequential()
features:add(SpatialConvolution(3,96,7,7,2,2,2,2))       -- 224 -> 111
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 110 -> 55
features:add(nn.SpatialBatchNormalization(0))
features:add(SpatialConvolution(96,256,5,5,2,2,1,1))       --  55 -> 27
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(nn.SpatialBatchNormalization(0))
features:add(SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
features:add(ReLU(true))
features:add(nn.SpatialBatchNormalization(0))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(ReLU(true))
features:add(nn.SpatialBatchNormalization(0))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
features:add(nn.SpatialBatchNormalization(0))

local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(0))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(0))
classifier:add(nn.Linear(4096, 1000))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()

function fillBias(m)
for i=1, #m.modules do
    if m:get(i).bias then
        m:get(i).bias:fill(0.1)
    end
end
end

fillBias(features)
fillBias(classifier)
model:add(features):add(classifier)

return model

