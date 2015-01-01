require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'eladtools'
require 'trepl'

----------------------------------------------------------------------

print '==> processing options'

opt = lapp[[
-r,--learningRate       (default 0.001)          learning rate
-c,--learningRateDecay  (default 1e-6)           learning rate decay (in # samples)
-w,--weightDecay        (default 1e-4)           L2 penalty on the weights
-m,--momentum           (default 0.9)            momentum
-b,--batchSize          (default 128)            batch size
-t,--threads            (default 8)              number of threads
-p,--type               (default cuda)           float or cuda
-i,--devid              (default 1)              device ID (if using CUDA)
-o,--save               (default results)        save directory
-n, --network	        (default CaffeRef_Model) Model - must return valid network. Available - {CaffeRef_Model, AlexNet_Model, NiN_Model, OverFeat_Model}
-v, --visualize         (default 0)              visualizing results
-z, --optimization      (default sgd)            optimization method
-e, --epoch             (default -1)             number of epochs to train -1 for unbounded
]]

torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)
----------------------------------------------------------------------
-- Model + Loss:
local model = require('./Models/' .. opt.network)
local loss = nn.ClassNLLCriterion()
local InputSize = model.InputSize or 224
-- classes
local data = require 'Data'
local classes = data.ImageNetClasses.ClassName

local ccn2_compatibility = true
----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Output files configuration
local weights_filename = paths.concat(opt.save, opt.network .. '_Weights.t7')
local log_filename = paths.concat(opt.save, opt.network .. '_TrainingLog.log')
os.execute('mkdir -p ' .. sys.dirname(log_filename))
local Log = optim.Logger(log_filename)

----------------------------------------------------------------------
local TensorType = 'torch.FloatTensor'
if opt.type =='cuda' then
    model:cuda()
    loss = loss:cuda()
    TensorType = 'torch.CudaTensor'
end

local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.learningRateDecay
}

-- Use random crops from the sample images
local function CropFunction(img,label)
    local nDim = img:dim()
    local start_x = math.random(img:size(nDim)-InputSize)
    local start_y = math.random(img:size(nDim-1)-InputSize)
    return img:narrow(nDim,start_x,InputSize):narrow(nDim-1,start_y,InputSize):add(-118.380948):div(61.896913),label
end

local function updateConfusion(y,yt)
    confusion:batchAdd(y,yt)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()
local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    OptFunction = optim.sgd,
    OptState = optimState,
    Parameters = {Weights, Gradients},
    HookFunction = updateConfusion
}


------------------------------
local function Train(Data)

    model:training()

    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        AutoLoad = true,
        MaxNumItems = opt.batchSize,
        Source = Data,
        ExtractFunction = CropFunction,
        TensorType = TensorType
    }

    local yt = MiniBatch.Labels
    local x = MiniBatch.Data
    local SizeData = Data.Source:size()
    local NumSamples = 0
    local NumBatches = 0
    while MiniBatch:GetNextBatch() do
        NumSamples = NumSamples+x:size(1)
        NumBatches = NumBatches + 1
        if ccn2_compatibility==false or math.fmod(x:size(1),32)==0 then
            optimizer:optimize(x, yt)
        end
        xlua.progress(NumSamples, SizeData)

        if math.fmod(NumBatches,50)==0 then
            collectgarbage()
        end
    end

end
------------------------------
local function Test(Data)

    model:evaluate()

    local MiniBatch = DataProvider{
        Name = 'GPU_Batch',
        AutoLoad = true,
        MaxNumItems = opt.batchSize,
        Source = Data,
        ExtractFunction = CropFunction,
        TensorType = TensorType
    }

    local yt = MiniBatch.Labels
    local x = MiniBatch.Data
    local SizeData = Data.Source:size()
    local NumSamples = 0
    local NumBatches = 0
    while MiniBatch:GetNextBatch() do
        NumSamples = NumSamples+x:size(1)
        NumBatches = NumBatches + 1
        if ccn2_compatibility==false or math.fmod(x:size(1),32)==0 then
            local y = model:forward(x)
            updateConfusion(y,yt)
        end
        xlua.progress(NumSamples, SizeData)

        if math.fmod(NumBatches,50)==0 then
            collectgarbage()
        end
    end

end

local epoch = 1
while true do
    data.TrainingData:Reset()
    data.ValidationData:Reset()

    print('Epoch ' .. epoch) 
    confusion:zero()
    Train(data.TrainingData)
    torch.save(weights_filename, Weights)
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    print(ErrTrain)

    confusion:zero()
    Test(data.ValidationData)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)

    print('Training Error = ' .. ErrTrain)
    print('Test Error = ' .. ErrTest)
    Log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
        Log:plot()
    end
    epoch = epoch+1
end
