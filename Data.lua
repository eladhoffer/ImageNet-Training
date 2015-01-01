require 'eladtools'
require 'image'
require 'xlua'
local gm = require 'graphicsmagick'
local ffi = require 'ffi'

-------------------------------Settings----------------------------------------------
local TRAINING_PATH = '/home/ehoffer/Datasets/ImageNet/ILSVRC2012_img_train/'
local VALIDATION_PATH = '/home/ehoffer/Datasets/ImageNet/ILSVRC2012_img_val/'
local ImageSize = 256--224
local LoadedImageSize = 256
local BatchSize = 8192--2560
-------------------------------------------------------------------------------------

local ImageNetClasses = torch.load('./ImageNetClasses')
local ValidationLabels = torch.load('./ValidationLabels')

for i=1001,#ImageNetClasses.ClassName do
    ImageNetClasses.ClassName[i] = nil
end
------------------------------------------Data Functions-----------------------------------------------
local LabelTraining = function(Item)
    local fn = paths.basename(Item,'JPEG')
    local wnid = string.split(fn,'_')[1]
    local ClassNum = ImageNetClasses.Wnid2ClassNum[wnid]
    return ClassNum
end

local LabelValidation = function(Item)
    local fn = paths.basename(Item,'JPEG')
    local num_img = tonumber(string.split(fn,'_')[3])
    return ValidationLabels[num_img]
end

local PreProcess = function(Img)
    --local im = PadTensor(CropCenter(Img,ImageSize),ImageSize,ImageSize)--:add(-0.45):mul(4)--:add(-118.380948):div(61.896913)--:add(-0.45):mul(4)

    local im = CropCenter(Img,ImageSize)--:add(-0.45):mul(4)--:add(-118.380948):div(61.896913)--:add(-0.45):mul(4)
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

local LoadImg = function(filename)
   return gm.Image(filename, LoadedImageSize):size(nil,LoadedImageSize):toTensor('byte','RGB','DHW')
   -- return image.load(filename,3,'byte')
end
------------------------------------------Training Data----------------------------------------------
local GetFromFilenameTraining = function(charTensor,_)
    local Images = torch.ByteTensor(charTensor:size(1), 3, ImageSize, ImageSize)
    local Labels = torch.LongTensor(charTensor:size(1))
    for i=1,charTensor:size(1) do
        local data = torch.data(charTensor[i])
        local filename = ffi.string(data)
        Labels[i] = LabelTraining(filename)
        local img = PreProcess(LoadImg(filename))
        if img:size(2)~= ImageSize or img:size(3) ~= ImageSize or img:size(1) ~= 3 or img:dim()~=3 then
            print(img:size())
        end
        Images[i] =img 
        xlua.progress(i,charTensor:size(1))
    end
    return Images, Labels
end


--------------------------------------------Validation Data----------------------------------------------
local GetFromFilenameValidation = function(charTensor,_)
    local Images = torch.FloatTensor(charTensor:size(1), 3, ImageSize, ImageSize)
    local Labels = torch.FloatTensor(charTensor:size(1))
    for i=1,charTensor:size(1) do
        local data = torch.data(charTensor[i])
        local filename = ffi.string(data)
        Labels[i] = LabelValidation(filename)
        local img = PreProcess(LoadImg(filename))
        if img == nil then
            print('Image is buggy')
            print(filename)
            Images[i] = Images[1]
        else
            Images[i] = img
        end
        xlua.progress(i,charTensor:size(1))
    end
    return Images, Labels
end

-------------- Getting Filenames ----------------------------
local VALIDATION_DIR = './Data/ValidationCache/'
local TRAINING_DIR = './Data/TrainingCache/'

local TrainingFiles = FileSearcher{
    Name = 'TrainingFilenames',
    CachePrefix = TRAINING_DIR,
    MaxNumItems = 1e7,
    CacheFiles = true,
    PathList = {TRAINING_PATH},
    Shuffle = false,--true,
    SubFolders = true
}
local ValidationFiles = FileSearcher{
    Name = 'ValidationFilenames',
    CachePrefix = VALIDATION_DIR,
    MaxNumItems = 1e7,
    CacheFiles = true,
    PathList = {VALIDATION_PATH}
}
----------------------------------------------------------------

local TrainingData = DataProvider{
    Name = 'TrainingData',
    CachePrefix = TRAINING_DIR,
    CacheFiles = true,
    Source = TrainingFiles,
    MaxNumItems = BatchSize,
    CopyData = false,
    TensorType = 'torch.ByteTensor',
    ExtractFunction = GetFromFilenameTraining,
    DataContainer = false

}
local ValidationData = DataProvider{
    Name = 'ValidationData',
    CachePrefix = VALIDATION_DIR,
    CacheFiles = true,
    Source = ValidationFiles,
    MaxNumItems = BatchSize,
    CopyData = false,
    TensorType = 'torch.ByteTensor',
    ExtractFunction = GetFromFilenameValidation,
    DataContainer = false

}

--------------------------------------------Returned Values----------------------------------------------
return{
    TrainingData = TrainingData,
    ValidationData = ValidationData,
    ImageNetClasses = ImageNetClasses
}
