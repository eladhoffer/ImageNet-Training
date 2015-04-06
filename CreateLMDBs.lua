require 'eladtools'
require 'image'
require 'xlua'
require 'lmdb'

local gm = require 'graphicsmagick'
local ffi = require 'ffi'
local config = require 'Config'


-------------------------------Settings----------------------------------------------

local PreProcess = function(Img)
    local im = CropCenter(Img,config.ImageSize)
    if im:dim() == 2 then
        im = im:reshape(1,im:size(1),im:size(2))
    end
    if im:size(1) == 1 then
        im=torch.repeatTensor(im,3,1,1)
    end
    if im:size(1) > 3 then
        im = im[{{1,3},{},{}}]
    end
    return im
end


local LoadImgData = function(filename)
    local img = gm.Image(filename, config.ImageSize):size(nil,config.ImageSize):toTensor('byte','RGB','DHW')

    if img == nil then
        print('Image is buggy')
        print(filename)
        os.exit()
    end 
    return PreProcess(img)---image.compressJPG(img)
end

function NameFile(filename)
    local name = paths.basename(filename,'JPEG')
    local substring = string.split(name,'_')

    if substring[1] == 'ILSVRC2012' then -- Validation file
        local num = tonumber(substring[3])
        return config.ImageNetClasses.ClassNum2Wnid[config.ValidationLabels[num]] .. '_' .. num
    else -- Training file
        return name
    end

end

function LMDBFromFilenames(charTensor,env)
    env:open()
    local txn = env:txn()
    local cursor = txn:cursor()
    for i=1,charTensor:size(1) do
        local filename = ffi.string(torch.data(charTensor[i]))
        local data = {Data = LoadImgData(filename), Name = NameFile(filename)}
        
        cursor:put(string.format('%07d',i),data, lmdb.C.MDB_NODUPDATA) 
        if i % 1000 == 0 then
            txn:commit()
            print(env:stat())
            collectgarbage()
            txn = env:txn()
            cursor = txn:cursor()
        end
        xlua.progress(i,charTensor:size(1))
    end
    txn:commit()
    env:close()

end


local TrainingFiles = FileSearcher{
    Name = 'TrainingFilenames',
    CachePrefix = config.TRAINING_DIR,
    MaxNumItems = 1e8,
    CacheFiles = true,
    PathList = {config.TRAINING_PATH},
    SubFolders = true
}
local ValidationFiles = FileSearcher{
    Name = 'ValidationFilenames',
    CachePrefix = config.VALIDATION_DIR,
    MaxNumItems = 1e8,
    PathList = {config.VALIDATION_PATH}
}

local TrainDB= lmdb.env{
    Path = config.TRAINING_DIR,
    Name = 'TrainDB'
}

local ValDB= lmdb.env{
    Path = config.VALIDATION_DIR,
    Name = 'ValDB'
}

TrainingFiles:ShuffleItems()
LMDBFromFilenames(TrainingFiles.Data, TrainDB)
LMDBFromFilenames(ValidationFiles.Data, ValDB)


