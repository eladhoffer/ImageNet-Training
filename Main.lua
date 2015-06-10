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
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'AlexNet',              'Model file - must return valid network.')
cmd:option('-LR',                 0.01,                   'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        5e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-bufferSize',         1280,                   'buffer size')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')
cmd:option('-checkpoint',         0,                      'Save a weight check point every n samples. 0 for off')

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                  'shuffle training samples')


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
-- classes
local config = require 'Config'
config.InputSize = model.InputSize or 224

local data = require 'Data'
local classes = data.ImageNetClasses.ClassName

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local AllowVarBatch = true
for _,m in pairs(model.modules) do
    if torch.type(m):find('ccn2') then
        AllowVarBatch = false
        break
    end
end


----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local weightsFilename = paths.concat(opt.save, 'modelWeights')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------
print '==> Network'
print(model)
print '==> Loss'
print(loss)

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
    AllowVarBatch = false
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()
print(Weights:nElement() ..  ' Parameters')

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('==>Loaded Weights from: ' .. opt.load)
    Weights:copy(w)  
end
if opt.nGPU > 1 then
    model:syncParameters()
end

--------------Optimization Configuration--------------------------

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

------------------------------
local function Forward(DB, train)
    confusion:zero()

    local SizeData = DB:size()
    if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end
    local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()
    if train then --shuflle batches from LMDB 
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
        if currBatch >= dataIndices:size(1) then return end 
        currBuffer = currBuffer%numBuffers +1
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
            confusion:batchAdd(y,yt)
            NumSamples = NumSamples+x:size(1)
            xlua.progress(NumSamples, SizeData)
        end
        if train and opt.checkpoint >0 and NumSamples % opt.checkpoint == 0 then
            confusion:updateValids()
            print('\nAfter ' .. NumSamples .. ' samples, current error is: ' .. 1-confusion.totalValid .. '\n')
            torch.save(weightsFilename .. '_checkpoint' .. '.t7', savedModel)
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
local epoch = 1
while true do
    print('\nEpoch ' .. epoch) 
    local LossTrain = Train(data.TrainDB)
    torch.save(weightsFilename .. '_' .. epoch .. '.t7', savedModel)
    torch.save(optStateFilename .. '_' .. epoch .. '.t7', optimState)
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    print('\nTraining Loss: ' .. LossTrain)
    print('Training Classification Error: ' .. ErrTrain)

    local LossVal = Test(data.ValDB)
    confusion:updateValids()
    local ErrVal = (1-confusion.totalValid)


    print('\nValidation Loss: ' .. LossVal)
    print('Validation Classification Error = ' .. ErrVal)
    Log:add{['Training Error']= ErrTrain, ['Validation Error'] = ErrVal}

    Log:style{['Training Error'] = '-', ['Validation Error'] = '-'}
    Log:plot()

    epoch = epoch+1
end
