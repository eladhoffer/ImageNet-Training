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

function Normalize(data, label)
    return data:float():add(-config.DataMean):div(config.DataStd)
end



local nthread = 1


local threads = Threads(nthread,
function()
    require 'lmdb'
    require 'eladtools'
    require 'util'
    --lmdb.verbose = false
end,

function()
    ImageNetClasses = config.ImageNetClasses
    SampleSize = config.SampleSize

    function LabelFromKey(Item)
        local wnid = string.split(Item,'_')[1]
        local ClassNum = config.ImageNetClasses.Wnid2ClassNum[wnid]
        return ClassNum
    end

    function ExtractFromLMDB(key, data)
        local class = LabelFromKey(data.Name)
        return data.Data, class
    end

    function CacheRand(env, locations)
        local num
        if type(locations) == 'table' then
            num = #locations
        else
            num = locations:size(1)
        end
        local txn = env:txn(true)
        local Data = torch.ByteTensor(num ,unpack(config.SampleSize))
        local Labels = torch.LongTensor(num)
        for i = 1, num do
            local data = txn:get(self.Keys[i])
            Data[i], Labels[i] = ExtractFromLMDB(locations[i], data)
        end
        txn:abort()
        return Data, Labels
    end

    function CacheSeq(env, start_pos, num)
        local num = num or 1
        local txn = env:txn(true)
        local cursor = txn:cursor()
        cursor:set(string.format('%07d',start_pos))

        local Data = torch.ByteTensor(num ,unpack(SampleSize))
        local Labels = torch.LongTensor(num)
        for i = 1, num do
            local key, data = cursor:get()
            if key == nil or data == nil then
                print(i+start_pos-1)
            end
            Data[i], Labels[i] = ExtractFromLMDB(key, data)
            cursor:next()
        end
        cursor:close()
        txn:abort()
        return Data, Labels
    end


end
)


local ValDB = lmdb.env({Path = config.VALIDATION_DIR, RDONLY = true})
local TrainDB = lmdb.env({Path = config.TRAINING_DIR, RDONLY = true})


return {
    ImageNetClasses = config.ImageNetClasses,
    ValDB = ValDB,
    TrainDB = TrainDB,
    ImageSize = config.ImageSize,
    Threads = threads
}
