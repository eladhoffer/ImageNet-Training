require 'nn'
require 'cudnn'

local InputMaps = 3
local InputWidth = 221
local InputHeight = 221

local KernelSize = {7,7,3,3,3,3,1,1,1,1}
local ConvStride = {2,1,1,1,1,1,1,1,1,1}
local Padding = {0,0,1,1,1,1,0,0,0,0}
local PoolSize =   {3,2,1,1,1,3,1,1,1,1}
local PoolStride= PoolSize
local TValue = 0
local TReplace = 0
local Outputs = 1000
local FeatMaps = {InputMaps, 96,256,512,512,1024,1024,4096,4096, Outputs}

local LayerNum

--------------Calculate size of feature maps - useful for linear layer flattening------------------------
SizeMap = {InputWidth}
for i=2, #FeatMaps do
    SizeMap[i] = math.floor(math.ceil((SizeMap[i-1] - KernelSize[i-1] + 1 + 2*Padding[i-1]) / ConvStride[i-1]) / PoolStride[i-1])
end

----------------Create Model-------------------------------------
model = nn.Sequential()

---------------Layer - Convolution + Max Pooling------------------
LayerNum = 1
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))
model:add(cudnn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))


---------------Layer - Convolution + Max Pooling------------------
LayerNum = 2
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))
model:add(cudnn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))

---------------Layer - Convolution ------------------
LayerNum = 3
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))


---------------layer - convolution ------------------
LayerNum = 4
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))


---------------layer - convolution ------------------
LayerNum = 5
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))


---------------Layer - Convolution + Max Pooling------------------
LayerNum = 6
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(cudnn.SpatialConvolution(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.Threshold(TValue, TReplace))
model:add(cudnn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))

---------------Layer - Fully connected ------------------
LayerNum = 7
model:add(nn.View(SizeMap[LayerNum]*SizeMap[LayerNum]*FeatMaps[LayerNum]))
model:add(nn.Linear(SizeMap[LayerNum]*SizeMap[LayerNum]*FeatMaps[LayerNum],  FeatMaps[LayerNum+1]))
model:add(nn.Threshold(TValue, TReplace))

---------------Layer - Fully connected ------------------
LayerNum = 8
model:add(nn.Dropout(0.5))
model:add(nn.Linear(FeatMaps[LayerNum], FeatMaps[LayerNum+1]))
model:add(nn.Threshold(TValue, TReplace))

---------------Layer - Fully connected classifier ------------------
LayerNum = 9
model:add(nn.Linear(FeatMaps[LayerNum], FeatMaps[LayerNum+1]))


---------------Layer - Log Probabilities--------------------------
model:add(nn.LogSoftMax())




--
--
--local finput = model.modules[1].finput
--local fgradInput = model.modules[1].fgradInput
--for i,layer in ipairs(model.modules) do
--    if layer.bias then
--        layer.bias:fill(.01)
--    end
--    if layer.finput then
--        layer.finput = finput
--    end
--    if layer.fgradInput then
--        layer.fgradInput = fgradInput
--    end
--end
--
--
--model.InputSize = InputWidth
return model
