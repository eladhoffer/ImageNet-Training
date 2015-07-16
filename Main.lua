require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'

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
cmd:option('-epoch',              -1,                       'number of epochs to train, -1 for unbounded')
cmd:option('-testonly',           false,                    'Just test loaded net on validation set')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                        'number of threads')
cmd:option('-type',               'cuda',                   'float or cuda')
cmd:option('-bufferSize',         1280,                     'buffer size')
cmd:option('-devid',              1,                        'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                        'num of gpu devices used')
cmd:option('-constBatchSize',     false,                    'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                       'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),   'save directory')
cmd:option('-optState',           false,                    'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                        'Save a weight check point every n samples. 0 for off')

cmd:text('===>Data Options')
cmd:option('-shuffle',            true,                     'shuffle training samples')


opt = cmd:parse(arg or {})
opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Model + Loss:
local model = require(opt.network)

local loss = nn.ClassNLLCriterion()

if torch.type(model) == 'table' then
    if model.loss then
        loss = model.loss
    end
    model = model.model
end

if paths.filep(opt.load) then
    model = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
end
-- classes
local config = require 'Config'
config.InputSize = model.InputSize or 224

local data = require 'Data'
local classes = data.ImageNetClasses.ClassName

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)


local AllowVarBatch = not opt.constBatchSize


----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
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
    model:syncParameters()
    savedModel = model.modules[1]:clone('weight','bias','running_mean','running_std')
else
    savedModel = model:clone('weight','bias','running_mean','running_std')
end

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)


------------------Optimization Configuration--------------------------

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
}

----------------------------------------------------------------------
local function ExtractSampleFunc(data, label)
    return Normalize(data),label
end

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
        BufferSources[i] = DataProvider{
            Source = {torch.ByteTensor(),torch.IntTensor()}
        }
    end


    local currBatch = 1

    local BufferNext = function()
        currBuffer = currBuffer%numBuffers +1
        if currBatch > dataIndices:size(1) then BufferSources[currBuffer] = nil return end
        local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch]+1)
        BufferSources[currBuffer].Data:resize(sizeBuffer ,unpack(config.SampleSize))
        BufferSources[currBuffer].Labels:resize(sizeBuffer)
        DB:AsyncCacheSeq(config.Key(dataIndices[currBatch]), sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
        currBatch = currBatch + 1
    end

    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = BufferSources[currBuffer],
        ExtractFunction = ExtractSampleFunc,
        TensorType = TensorType
    }


    local yt = MiniBatch.Labels
    local y = torch.Tensor()
    local x = MiniBatch.Data
    local NumSamples = 0
    local loss_val = 0
    local currLoss = 0

    BufferNext()

    while NumSamples < SizeData do

        DB:Synchronize()
        MiniBatch:Reset()
        MiniBatch.Source = BufferSources[currBuffer]
        if train and opt.shuffle then MiniBatch.Source:ShuffleItems() end
        BufferNext()


        while MiniBatch:GetNextBatch() do
            if train then
                if opt.nGPU > 1 then
                    model:zeroGradParameters()
                    model:syncParameters()
                end
                y, currLoss = optimizer:optimize(x, yt)
            else
                y = model:forward(x)
                currLoss = loss:forward(y,yt)
            end
            loss_val = currLoss + loss_val
            if type(y) == 'table' then --table results - always take first prediction
                y = y[1]
            end
            confusion:batchAdd(y,yt)
            NumSamples = NumSamples+x:size(1)
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
    return(loss_val/math.ceil(SizeData/opt.batchSize))
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
data.ValDB:Threads()
data.TrainDB:Threads()


if opt.testonly then opt.epoch = 2 end
local epoch = 1

while epoch ~= opt.epoch do
    local ErrTrain, LossTrain
    if not opt.testonly then
        print('\nEpoch ' .. epoch)
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

    epoch = epoch+1
end
