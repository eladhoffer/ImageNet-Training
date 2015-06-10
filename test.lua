require 'eladtools'
require 'lmdb'
require 'image'
local config = require 'Config'

ImageNetClasses = config.ImageNetClasses
SampleSize = config.SampleSize
config.InputSize = config.InputSize or 224
-- Use random crop from the sample image
function ExtractFromLMDBTrain(key, data)
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = image.decompressJPG(data.Data):float()
    local nDim = img:dim()
    local start_x = math.random(img:size(nDim)-config.InputSize)
    local start_y = math.random(img:size(nDim-1)-config.InputSize)
    img = img:narrow(nDim,start_x,config.InputSize):narrow(nDim-1,start_y,config.InputSize)

    return img, class
end

function ExtractFromLMDBTest(key, data)
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = image.decompressJPG(data.Data):float()
    local nDim = img:dim()
    local start_x = math.ceil((img:size(nDim)-config.InputSize)/2)
    local start_y = math.ceil((img:size(nDim-1)-config.InputSize)/2)
    img = img:narrow(nDim,start_x,config.InputSize):narrow(nDim-1,start_y,config.InputSize)

    return img, class
end

 TrainDB = eladtools.LMDBProvider{
     Source = lmdb.env({Path = config.VALIDATION_DIR , RDONLY = true}),
    SampleSize = config.SampleSize,
    ExtractFunction = ExtractFromLMDBTrain
}

function Keys(tensor)
    local tbl = {}
    for i=1,tensor:size(1) do
        tbl[i] = config.Key(tensor[i])
    end
    return tbl
end

print(TrainDB:size())
print(config.SampleSize)

local DataSetSize = TrainDB:size()
local Num = 128
x = torch.FloatTensor(Num ,unpack(config.SampleSize))
y = torch.IntTensor(Num)

--local t
--t=torch.tic()
--TrainDB:CacheSeq(Key(math.random(DataSetSize-Num)),Num ,x,y)
--t=torch.tic()-t
--print('Sequential Time: ' .. t/Num .. ' Per sample')
--
--
--t=torch.tic()
--TrainDB:CacheRand(Keys(torch.randperm(DataSetSize):narrow(1,1,Num)),x,y)
--t=torch.tic()-t
--print('Random Access Time: ' .. t/Num .. ' Per sample')

TrainDB:Threads()
t=torch.tic()
TrainDB:AsyncCacheSeq(Key(1),Num ,x,y)
t=torch.tic()-t
print(x[Num]:mean())
print('Async Random Access Time: ' .. t/Num .. ' Per sample')
print(x[Num]:mean())
