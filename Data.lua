require 'xlua'
require 'lmdb'


local DataProvider = require 'DataProvider'
local config = require 'Config'


function ExtractFromLMDBTrain(data)
    require 'image'
    local reSample = function(sampledImg)
        local sizeImg = sampledImg:size()
        local szx = torch.random(math.ceil(sizeImg[3]/4))
        local szy = torch.random(math.ceil(sizeImg[2]/4))
        local startx = torch.random(szx)
        local starty = torch.random(szy)
        return image.scale(sampledImg:narrow(2,starty,sizeImg[2]-szy):narrow(3,startx,sizeImg[3]-szx),sizeImg[3],sizeImg[2])
    end
    local rotate = function(angleRange)
        local applyRot = function(Data)
            local angle = torch.randn(1)[1]*angleRange
            local rot = image.rotate(Data,math.rad(angle),'bilinear')
            return rot
        end
        return applyRot
    end

    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end

    if math.min(img:size(2), img:size(3)) ~= config.ImageMinSide then
        img = image.scale(img, '^' .. config.ImageMinSide)
    end

    if config.Augment == 3 then
        img = rotate(0.1)(img)
        img = reSample(img)
    elseif config.Augment == 2 then
        img = reSample(img)
    end
    local startX = math.random(img:size(3)-config.InputSize[3]+1)
    local startY = math.random(img:size(2)-config.InputSize[2]+1)

    img = img:narrow(3,startX,config.InputSize[3]):narrow(2,startY,config.InputSize[2])
    local hflip = torch.random(2)==1
    if hflip then
        img = image.hflip(img)
    end

    return img, class
end

function ExtractFromLMDBTest(data)
    require 'image'
    local wnid = string.split(data.Name,'_')[1]
    local class = config.ImageNetClasses.Wnid2ClassNum[wnid]
    local img = data.Data
    if config.Compressed then
        img = image.decompressJPG(img,3,'byte')
    end

    if (math.min(img:size(2), img:size(3)) ~= config.ImageMinSide) then
        img = image.scale(img, '^' .. config.ImageMinSide)
    end

    local startX = math.ceil((img:size(3)-config.InputSize[3]+1)/2)
    local startY = math.ceil((img:size(2)-config.InputSize[2]+1)/2)
    img = img:narrow(3,startX,config.InputSize[3]):narrow(2,startY,config.InputSize[2])
    return img, class
end

function Keys(tensor)
    local tbl = {}
    for i=1,tensor:size(1) do
        tbl[i] = config.Key(tensor[i])
    end
    return tbl
end

function EstimateMeanStd(DB, typeVal, numEst)
    local typeVal = typeVal or 'simple'
    local numEst = numEst or 10000
    local x = torch.FloatTensor(numEst ,unpack(config.InputSize))
    local randKeys = Keys(torch.randperm(DB:size()):narrow(1,1,numEst))
    DB:CacheRand(randKeys, x)
    local dp = DataProvider.Container{
        Source = {x, nil}
    }
    return {typeVal, dp:normalize(typeVal)}
end

local TrainDB = DataProvider.LMDBProvider{
    Source = lmdb.env({Path = config.TRAINING_DIR, RDONLY = true}),
    ExtractFunction = ExtractFromLMDBTrain
}
local ValDB = DataProvider.LMDBProvider{
    Source = lmdb.env({Path = config.VALIDATION_DIR , RDONLY = true}),
    ExtractFunction = ExtractFromLMDBTest
}



return {
    ImageNetClasses = config.ImageNetClasses,
    ValDB = ValDB,
    TrainDB = TrainDB,
}
