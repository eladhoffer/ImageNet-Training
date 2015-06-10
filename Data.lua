require 'eladtools'
require 'xlua'
require 'util'
require 'lmdb'

local Threads = require 'threads'
local ffi = require 'ffi'
local config = require 'Config'

-- Use random crop from the sample image
function CropRandPatch(img, InputSize)
    local nDim = img:dim()
    local start_x = math.random(img:size(nDim)-InputSize)
    local start_y = math.random(img:size(nDim-1)-InputSize)
    return img:narrow(nDim,start_x,InputSize):narrow(nDim-1,start_y,InputSize)
end

function CropCenterPatch(img, InputSize)
    local nDim = img:dim()
    local start_x = math.ceil((img:size(nDim)-InputSize)/2)
    local start_y = math.ceil((img:size(nDim-1)-InputSize)/2)
    return img:narrow(nDim,start_x,InputSize):narrow(nDim-1,start_y,InputSize)
end

function Normalize(data)
    return data:float():add(-config.DataMean):div(config.DataStd)
end
function ExtractFromLMDBTrain(key, data)
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end
    local nDim = img:dim()
    local start_x = math.random(img:size(nDim)-config.InputSize)
    local start_y = math.random(img:size(nDim-1)-config.InputSize)
    img = img:narrow(nDim,start_x,config.InputSize):narrow(nDim-1,start_y,config.InputSize)
    
    local hflip = math.random(2)==1
    if hflip then
        img = image.hflip(img)
    end

    return img, class
end

function ExtractFromLMDBTest(key, data)
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end
    local nDim = img:dim()
    local start_x = math.ceil((img:size(nDim)-config.InputSize)/2)
    local start_y = math.ceil((img:size(nDim-1)-config.InputSize)/2)
    img = img:narrow(nDim,start_x,config.InputSize):narrow(nDim-1,start_y,config.InputSize)

    return img, class
end

function Keys(tensor)
    local tbl = {}
    for i=1,tensor:size(1) do
        tbl[i] = config.Key(tensor[i])
    end
    return tbl
end

local TrainDB = eladtools.LMDBProvider{
    Source = lmdb.env({Path = config.TRAINING_DIR, RDONLY = true}),
    SampleSize = config.SampleSize,
    ExtractFunction = ExtractFromLMDBTrain
}
local ValDB = eladtools.LMDBProvider{
    Source = lmdb.env({Path = config.VALIDATION_DIR , RDONLY = true}),
    SampleSize = config.SampleSize,
    ExtractFunction = ExtractFromLMDBTest
}



return {
    ImageNetClasses = config.ImageNetClasses,
    ValDB = ValDB,
    TrainDB = TrainDB,
}
