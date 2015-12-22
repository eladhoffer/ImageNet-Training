require 'torch'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'
local DataProvider = require 'DataProvider'

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a network on ILSVRC ImageNet task')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',              'Models Folder')
cmd:option('-network',            'AlexNet',                'Model file - must return valid network.')
cmd:option('-LR',                 0.01,                     'learning rate')
cmd:option('-LRDecay',            0,                        'learning rate decay (in # samples)')
cmd:option('-weightDecay',        5e-4,                     'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                      'momentum')
cmd:option('-batchSize',          128,                      'batch size')
cmd:option('-optimization',       'sgd',                    'optimization method')
cmd:option('-seed',               123,                      'torch manual random number generator seed')
cmd:option('-epoch',              -1,                       'number of epochs to train, -1 for unbounded')
cmd:option('-testonly',           false,                    'Just test loaded net on validation set')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                        'number of threads')
cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-bufferSize',         5120,                     'buffer size')
cmd:option('-devid',              1,                        'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                        'num of gpu devices used')
cmd:option('-constBatchSize',     false,                    'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                       'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),   'save directory')
cmd:option('-optState',           false,                    'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                        'Save a weight check point every n samples. 0 for off')

cmd:text('===>Data Options')
cmd:option('-augment',            1,                        'data augmentation level - {1 - simple mirror and crops, 2 +scales, 3 +rotations}')
cmd:option('-estMeanStd',         'preDef',                 'estimate mean and std. Options: {preDef, simple, channel, image}')
cmd:option('-shuffle',            true,                     'shuffle training samples')


opt = cmd:parse(arg or {})
opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.type == 'cuda' then
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
end

----------------------------------------------------------------------
local config = require 'Config'

-- Model + Loss:
local model = require(opt.network)
local loss = model.loss or nn.ClassNLLCriterion()
local trainRegime = model.regime
local normalization = model.normalization or config.Normalization

config.InputSize = model.inputSize or {3, 224, 224}
config.ImageMinSide = model.rescaleImgSize or config.ImageMinSide

model = model.model or model --case of table model

if paths.filep(opt.load) then
    model = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
end

local data = require 'Data'

-- classes
local classes = data.ImageNetClasses.ClassName

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local AllowVarBatch = not opt.constBatchSize

----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)

cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------

local TensorType = 'torch.FloatTensor'
if opt.type =='cuda' then
    model:cuda()
    loss = loss:cuda()
    TensorType = 'torch.CudaTensor'
end



---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

local savedModel --savedModel - lower footprint model to save
if opt.nGPU > 1 then
    savedModel = model.modules[1]:clone('weight','bias','running_mean','running_std')
else
    savedModel = model:clone('weight','bias','running_mean','running_std')
end

----------------------------------------------------------------------
if opt.estMeanStd ~= 'preDef' then
  normalization = EstimateMeanStd(data.TrainDB, opt.estMeanStd)
end

if #normalization>0 then
  print '\n==> Normalization'
  if normalization[1] == 'simple' or normalization[1] == 'channel' then
    print(unpack(normalization))
  else
    print(normalization[1])
  end
end

print '\n==> Network'
print(model)
print('\n==>' .. Weights:nElement() ..  ' Parameters')

print '\n==> Loss'
print(loss)

if trainRegime then
    print '\n==> Training Regime'
    table.foreach(trainRegime, function(x, val) print(string.format('%012s',x), unpack(val)) end)
end


------------------Optimization Configuration--------------------------

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay,
    dampening = 0
}

local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
    Regime = trainRegime
}

----------------------------------------------------------------------
local function Forward(DB, train)
    confusion:zero()

    local SizeData = DB:size()
    if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end
    local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()
    if train and opt.shuffle then --shuffle batches from LMDB
        dataIndices = dataIndices:index(1, torch.randperm(dataIndices:size(1)):long())
    end

    local numBuffers = 2
    local currBuffer = 1
    local BufferSources = {}
    for i=1,numBuffers do
        BufferSources[i] = DataProvider.Container{
            Source = {torch.ByteTensor(),torch.IntTensor()}
        }
    end


    local currBatch = 1

    local BufferNext = function()
        currBuffer = currBuffer%numBuffers +1
        if currBatch > dataIndices:size(1) then BufferSources[currBuffer] = nil return end
        local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch]+1)
        BufferSources[currBuffer].Data:resize(sizeBuffer ,unpack(config.InputSize))
        BufferSources[currBuffer].Labels:resize(sizeBuffer)
        DB:asyncCacheSeq(config.Key(dataIndices[currBatch]), sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
        currBatch = currBatch + 1
    end

    local MiniBatch = DataProvider.Container{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = BufferSources[currBuffer],
        TensorType = TensorType
    }


    local yt = MiniBatch.Labels
    local y = torch.Tensor()
    local x = MiniBatch.Data
    local NumSamples = 0
    local lossVal = 0
    local currLoss = 0

    BufferNext()

    while NumSamples < SizeData do
        DB:synchronize()
        MiniBatch:reset()
        MiniBatch.Source = BufferSources[currBuffer]
        if train and opt.shuffle then MiniBatch.Source:shuffleItems() end
        BufferNext()

        while MiniBatch:getNextBatch() do
            if #normalization>0 then MiniBatch:normalize(unpack(normalization)) end
            if train then
                y, currLoss = optimizer:optimize(x, yt)
                if opt.nGPU > 1 then
                    model:syncParameters()
                end
            else
                y = model:forward(x)
                currLoss = loss:forward(y,yt)
            end
            lossVal = currLoss + lossVal
            if type(y) == 'table' then --table results - always take first prediction
                y = y[1]
            end
            confusion:batchAdd(y,yt)
            NumSamples = NumSamples + x:size(1)
            xlua.progress(NumSamples, SizeData)
        end

        if train and opt.checkpoint >0 and (currBatch % math.ceil(opt.checkpoint/opt.bufferSize) == 0) then
            print(NumSamples)
            confusion:updateValids()
            print('\nAfter ' .. NumSamples .. ' samples, current error is: ' .. 1-confusion.totalValid .. '\n')
            torch.save(netFilename .. '_checkpoint' .. '.t7', savedModel)
        end
        collectgarbage()
    end
    xlua.progress(NumSamples, SizeData)
    return(lossVal/math.ceil(SizeData/opt.batchSize))
end

local function Train(Data)
    model:training()
    return Forward(Data, true)
end

local function Test(Data)
    model:evaluate()
    return Forward(Data, false)
end

------------------------------
data.ValDB:threads()
data.TrainDB:threads()


if opt.testonly then opt.epoch = 2 end
local epoch = 1

while epoch ~= opt.epoch do
    local ErrTrain, LossTrain
    if not opt.testonly then
        print('\nEpoch ' .. epoch ..'\n')
        optimizer:updateRegime(epoch, true)
        LossTrain = Train(data.TrainDB)
        torch.save(netFilename .. '_' .. epoch .. '.t7', savedModel)
        if opt.optState then
            torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
        end
        confusion:updateValids()
        ErrTrain = (1-confusion.totalValid)
        print('\nTraining Loss: ' .. LossTrain)
        print('Training Classification Error: ' .. ErrTrain)
    end

    local LossVal = Test(data.ValDB)
    confusion:updateValids()
    local ErrVal = (1-confusion.totalValid)


    print('\nValidation Loss: ' .. LossVal)
    print('Validation Classification Error = ' .. ErrVal)

    if not opt.testonly then
        Log:add{['Training Error']= ErrTrain, ['Validation Error'] = ErrVal}
        Log:style{['Training Error'] = '-', ['Validation Error'] = '-'}
        Log:plot()
    end

    epoch = epoch + 1
end
