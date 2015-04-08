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
cmd:option('-LR',                 0.01,                  'learning rate')
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

cmd:text('===>Data Options')
cmd:option('-shuffle',            false,                  'shuffle training samples')
cmd:option('-augment',            true,                   'Augment training data')


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
local data = require 'Data'
local threads = data.Threads
local classes = data.ImageNetClasses.ClassName

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local InputSize = model.InputSize or 224
local ccn2_compatibility = false
for _,m in pairs(model.modules) do
    if torch.type(m):find('ccn2') then
        ccn2_compatibility = true
        break
    end
end

----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local weights_filename = paths.concat(opt.save, 'Weights')
local log_filename = paths.concat(opt.save,'ErrorRate.log')
local Log = optim.Logger(log_filename)
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
    ccn2_compatibility = true
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

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
local function ExtractSampleFunc(augment)
    local f
    if augment then
        f = function(data,label)
            return Normalize(CropRandPatch(data, InputSize)), label
        end
    else
        f = function(data,label)
            return Normalize(CropCenterPatch(data, InputSize)), label
        end
    end
    return f
end

------------------------------
local function Forward(DB, train)
    confusion:zero()

    DB:open()
    local SizeData = DB:stat()['entries']
    DB:close()
    local randBatches = torch.range(1, SizeData, opt.bufferSize):long()
    randBatches = randBatches:index(1, torch.randperm(randBatches:size(1)):long())

    local current_buffer = 1
    local data_buffer = {torch.ByteTensor(), torch.ByteTensor()}
    local labels_buffer = {torch.LongTensor(), torch.LongTensor()}
    
    local buffer = function(start)
        if train and opt.shuffle then   -- Pseudo shuffling by skipping to random batch
           start = randBatches[math.ceil(start/opt.bufferSize)]
        end

        local num = math.min(SizeData - start, opt.bufferSize)
        threads:addjob(
        function()
            DB:open()
            local data, labels = CacheSeq(DB, start, num)
            DB:close()
            return sendTensor(data), sendTensor(labels)
        end,

        function(data, labels)
            receiveTensor(data, data_buffer[current_buffer])
            receiveTensor(labels, labels_buffer[current_buffer])
        end
        )
    end
    local BufferSource = DataProvider{
        Source = {data_buffer[current_buffer], labels_buffer[current_buffer]}
    }
    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = BufferSource,
        ExtractFunction = ExtractSampleFunc(train and opt.augment), 
        TensorType = TensorType
    }


    local yt = MiniBatch.Labels
    local y = torch.Tensor()
    local x = MiniBatch.Data
    local NumSamples = 0
    local loss_val = 0
    local curr_loss = 0

    while NumSamples < SizeData do
        if NumSamples == 0 then
            buffer(NumSamples + 1)
        end

        threads:synchronize()
        
        BufferSource:LoadFrom(data_buffer[current_buffer], labels_buffer[current_buffer])
        current_buffer = current_buffer%2 +1
        MiniBatch:Reset()
        buffer(NumSamples+1)
        while MiniBatch:GetNextBatch() do
            if ccn2_compatibility==false or math.fmod(x:size(1),32)==0 then
                if train then
                    if opt.nGPU > 1 then
                        model:zeroGradParameters()
                        model:syncParameters()
                    end

                    y, curr_loss = optimizer:optimize(x, yt)

                else
                    y = model:forward(x)
                    curr_loss = loss:forward(y,yt)
                end
                loss_val = curr_loss + loss_val
                confusion:batchAdd(y,yt)
            end
            xlua.progress(NumSamples, SizeData)
            NumSamples = NumSamples+x:size(1)
            collectgarbage()
        end

    end
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
local epoch = 1
while true do
    print('\nEpoch ' .. epoch) 
    local LossTrain = Train(data.TrainDB)
    torch.save(weights_filename .. '_' .. epoch .. '.t7', Weights)
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
