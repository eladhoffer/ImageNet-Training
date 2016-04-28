require 'nn'

local SpatialConvolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling

local function ConvBN(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    local module = nn.Sequential()
    module:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
    module:add(SpatialBatchNormalization(nOutputPlane,1e-3,nil,true))
    module:add(ReLU(true))
    return module
end

local function Tower(tbl)
    local module = nn.Sequential()
    for i=1, #tbl do
        module:add(tbl[i])
    end
    return module
end

local function Residual(m)
    local module = nn.Sequential()
    local cat = nn.ConcatTable():add(nn.Identity()):add(m)
    module:add(cat):add(nn.CAddTable())
    return module
end


local function Stem()
    local module = nn.Sequential()
    module:add(ConvBN(3,32,3,3,2,2,0,0))
    module:add(ConvBN(32,32,3,3,1,1,0,0))
    module:add(ConvBN(32,64,3,3,1,1,1,1))

    local cat1 = nn.Concat(2)
    cat1:add(SpatialMaxPooling(3,3,2,2,0,0))
    cat1:add(ConvBN(64,96,3,3,2,2,0,0))

    local cat2 = nn.Concat(2)
    cat2:add(Tower{
         ConvBN(160,64,1,1,1,1,0,0),
         ConvBN(64,64,7,1,1,1,3,0),
         ConvBN(64,64,1,7,1,1,0,3),
         ConvBN(64,96,3,3,1,1,0,0)
    })
    cat2:add(Tower{
         ConvBN(160,64,1,1,1,1,0,0),
         ConvBN(64,96,3,3,1,1,0,0)
    })

    local cat3 = nn.Concat(2)
    cat3:add(SpatialMaxPooling(3,3,2,2,0,0))
    cat3:add(ConvBN(192,192,3,3,2,2,0,0))

    module:add(cat1)
    module:add(cat2)
    module:add(cat3)

    return module
end

local function Reduction_A(nInputPlane,k,l,m,n)
    local module = nn.Concat(2)
    module:add(SpatialMaxPooling(3,3,2,2,0,0))
    module:add(ConvBN(nInputPlane,n,3,3,2,2,0,0))
    module:add(Tower{
        ConvBN(nInputPlane,k,1,1,1,1,0,0),
        ConvBN(k,l,3,3,1,1,1,1),
        ConvBN(l,m,3,3,2,2,0,0)
    })
    return module
end

local function Reduction_B(nInputPlane)
    local module = nn.Concat(2)
    module:add(SpatialMaxPooling(3,3,2,2,0,0))
    module:add(Tower{
        ConvBN(nInputPlane,256,1,1,1,1,0,0),
        ConvBN(256,384,3,3,2,2,0,0)
    })
    module:add(Tower{
        ConvBN(nInputPlane,256,1,1,1,1,0,0),
        ConvBN(256,256,3,3,2,2,0,0)
    })
    module:add(Tower{
        ConvBN(nInputPlane,256,1,1,1,1,0,0),
        ConvBN(256,288,3,3,1,1,1,1),
        ConvBN(288,256,3,3,2,2,0,0)
    })
    return module
end

local function Inception_ResNet_A(nInputPlane)
    local inceptionModule = nn.Concat(2)
    inceptionModule:add(ConvBN(nInputPlane,32,1,1,1,1,0,0))
    inceptionModule:add(Tower{
      ConvBN(nInputPlane,32,1,1,1,1,0,0),
      ConvBN(32,32,3,3,1,1,1,1)
    })
    inceptionModule:add(Tower{
      ConvBN(nInputPlane,32,1,1,1,1,0,0),
      ConvBN(32,48,3,3,1,1,1,1),
      ConvBN(48,64,3,3,1,1,1,1)
    })
    local preActivation = nn.Sequential()
    preActivation:add(ReLU(true))
    preActivation:add(inceptionModule)
    preActivation:add(SpatialConvolution(128,256,1,1,1,1,0,0))

    return Residual(preActivation)
end

local function Inception_ResNet_B(nInputPlane)
    local inceptionModule = nn.Concat(2)
    inceptionModule:add(ConvBN(nInputPlane,192,1,1,1,1,0,0))
    inceptionModule:add(Tower{
      ConvBN(nInputPlane,128,1,1,1,1,0,0),
      ConvBN(128,160,1,7,1,1,0,3),
      ConvBN(160,192,7,1,1,1,3,0)
    })
    local preActivation = nn.Sequential()
    preActivation:add(ReLU(true))
    preActivation:add(inceptionModule)
    preActivation:add(SpatialConvolution(384,896,1,1,1,1,0,0))

    return Residual(preActivation)
end

local function Inception_ResNet_C(nInputPlane)
    local inceptionModule = nn.Concat(2)
    inceptionModule:add(ConvBN(nInputPlane,192,1,1,1,1,0,0))
    inceptionModule:add(Tower{
      ConvBN(nInputPlane,192,1,1,1,1,0,0),
      ConvBN(192,224,1,3,1,1,0,1),
      ConvBN(224,256,3,1,1,1,1,0)
    })
    local preActivation = nn.Sequential()
    preActivation:add(ReLU(true))
    preActivation:add(inceptionModule)
    preActivation:add(SpatialConvolution(448,1792,1,1,1,1,0,0))

    return Residual(preActivation)
end


local model = Stem()
model:add(ConvBN(384,256,1,1,1,1,0,0))

for i=1,5 do
    model:add(Inception_ResNet_A(256))
end

model:add(Reduction_A(256,256,256,256,384))

for i=1,10 do
    model:add(Inception_ResNet_B(896))
end

model:add(Reduction_B(896))

for i=1,5 do
    model:add(Inception_ResNet_C(1792))
end

model:add(SpatialAveragePooling(8,8,1,1))
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Dropout(0.2))
model:add(nn.Linear(1792, 1000))
model:add(nn.LogSoftMax())

return {
  model = model,
  rescaleImgSize = 384,
  InputSize = {3, 299, 299}
}
