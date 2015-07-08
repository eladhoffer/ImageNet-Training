require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

local DimConcat = 2

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialMaxPooling = cudnn.SpatialMaxPooling
local SpatialAveragePooling = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU

local BNInception = false

---------------------------------------Inception Modules-------------------------------------------------
local Inception = function(nInput, n1x1, n3x3r, n3x3, dn3x3r, dn3x3, nPoolProj, type_pool,stride)
    local stride = stride or 1
    local InceptionModule = nn.Concat(DimConcat)

    if n1x1>0 then
        InceptionModule:add(nn.Sequential():add(SpatialConvolution(nInput,n1x1,1,1,stride,stride)))
    end

    if n3x3>0 and n3x3r>0 then
        local Module_3x3 = nn.Sequential()
        Module_3x3:add(SpatialConvolution(nInput,n3x3r,1,1)):add(ReLU(true))

        if BNInception then
            Module_3x3:add(nn.SpatialBatchNormalization(n3x3r,nil,nil,false))
        end

        Module_3x3:add(SpatialConvolution(n3x3r,n3x3,3,3,stride,stride,1,1))
        InceptionModule:add(Module_3x3)
    end

    if dn3x3>0 and dn3x3r>0 then
        local Module_d3x3 = nn.Sequential()
        Module_d3x3:add(SpatialConvolution(nInput,dn3x3r,1,1)):add(ReLU(true))

        if BNInception then
            Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r,nil,nil,false))
        end

        Module_d3x3:add(SpatialConvolution(dn3x3r,dn3x3r,3,3,1,1,1,1)):add(ReLU(true))

        if BNInception then
            Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r,nil,nil,false))
        end

        Module_d3x3:add(SpatialConvolution(dn3x3r,dn3x3,3,3,stride,stride,1,1))

        InceptionModule:add(Module_d3x3)
    end

    local PoolProj = nn.Sequential()
    if type_pool == 'avg' then
        PoolProj:add(SpatialAveragePooling(3,3,stride,stride,1,1))
    elseif type_pool == 'max' then
        PoolProj:add(SpatialMaxPooling(3,3,stride,stride,1,1))
    end
    if nPoolProj > 0 then
        PoolProj:add(SpatialConvolution(nInput, nPoolProj, 1, 1))
    end


    InceptionModule:add(PoolProj)
    return InceptionModule
end

-----------------------------------------------------------------------------------------------------------


local part1 = nn.Sequential()
part1:add(SpatialConvolution(3,64,7,7,2,2,3,3)) --3x224x224 -> 64x112x112
part1:add(SpatialMaxPooling(3,3,2,2):ceil()) -- 64x112x112 -> 64x56x56
part1:add(nn.SpatialBatchNormalization(64))
part1:add(ReLU(true))
part1:add(SpatialConvolution(64,192,3,3,1,1,1,1)) -- 64x56x56 -> 192x56x56
part1:add(SpatialMaxPooling(3,3,2,2):ceil()) -- 192x56x56 -> 192x28x28
part1:add(nn.SpatialBatchNormalization(192))
part1:add(ReLU(true))


--Inception(nInput, n1x1, n3x3r, n3x3, dn3x3r, dn3x3, nPoolProj, type_pool=['avg','max',nil])
part1:add(Inception(192,64,64,64,64,96,32,'avg')) --(3a) 192x28x28 -> 256x28x28
part1:add(nn.SpatialBatchNormalization(256))
part1:add(ReLU(true))
part1:add(Inception(256,64,64,96,64,96,64,'avg'))  --(3b) 256x28x28 -> 320x28x28
part1:add(nn.SpatialBatchNormalization(320))
part1:add(ReLU(true))
part1:add(Inception(320,0,128,160,64,96,0,'max',2)) --(3c) 320x28x28 -> 576x14x14
part1:add(nn.SpatialBatchNormalization(576))
part1:add(ReLU(true))

local part2 = nn.Sequential()
part2:add(Inception(576,224,64,96,96,128,128,'avg'))  --(4a) 576x14x14 -> 576x14x14
part2:add(nn.SpatialBatchNormalization(576))
part2:add(ReLU(true))
part2:add(Inception(576,192,96,128,96,128,128,'avg'))  --(4b) 576x14x14 -> 576x14x14
part2:add(nn.SpatialBatchNormalization(576))
part2:add(ReLU(true))
part2:add(Inception(576,160,128,160,128,160,96,'avg'))  --(4c) 576x14x14 -> 576x14x14
part2:add(nn.SpatialBatchNormalization(576,nil,nil,false))
part2:add(ReLU(true))

local part3 = nn.Sequential()
part3:add(Inception(576,96,128,192,160,192,96,'avg'))  --(4d) 576x14x14 -> 576x14x14
part3:add(nn.SpatialBatchNormalization(576))
part3:add(ReLU(true))
part3:add(Inception(576,0,128,192,192,256,0,'max',2))  --(4e) 576x14x14 -> 1024x7x7
part3:add(nn.SpatialBatchNormalization(1024,nil,nil,false))
part3:add(ReLU(true))

part3:add(Inception(1024,352,192,320,160,224,128,'avg'))  --(5a) 1024x7x7 -> 1024x7x7
part3:add(nn.SpatialBatchNormalization(1024))
part3:add(ReLU(true))
part3:add(Inception(1024,352,192,320,192,224,128,'max'))  --(5b) 1024x7x7 -> 1024x7x7
part3:add(nn.SpatialBatchNormalization(1024))
part3:add(ReLU(true))

--Classifier
local mainClassifier = nn.Sequential()
mainClassifier:add(nn.Dropout(0.5))
mainClassifier:add(SpatialAveragePooling(7,7))
mainClassifier:add(nn.View(1024):setNumInputDims(3))
mainClassifier:add(nn.Linear(1024,1000))
mainClassifier:add(nn.LogSoftMax())

local auxClassifier1 = nn.Sequential()
auxClassifier1:add(cudnn.SpatialAveragePooling(5,5,3,3):ceil())
auxClassifier1:add(cudnn.SpatialConvolution(576,128,1,1))
auxClassifier1:add(cudnn.ReLU(true))
auxClassifier1:add(nn.View(128*4*4):setNumInputDims(3))
auxClassifier1:add(nn.Linear(128*4*4,768))
auxClassifier1:add(cudnn.ReLU(true))
auxClassifier1:add(nn.Dropout(0.5))
auxClassifier1:add(nn.Linear(768,1000))
auxClassifier1:add(nn.LogSoftMax())

local auxClassifier2 = auxClassifier1:clone()

local input = nn.Identity()()
local output1 = part1(input)
local branch1 = auxClassifier1(output1)
local output2 = part2(output1)
local branch2 = auxClassifier2(output2)
local mainBranch = mainClassifier(part3(output2))
local model = nn.gModule({input},{mainBranch,branch1,branch2})

local NLL = nn.ClassNLLCriterion()
local loss = nn.ParallelCriterion(true):add(NLL):add(NLL,0.3):add(NLL,0.3)

return {model = model, loss = loss}
