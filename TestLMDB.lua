require 'eladtools'
require 'lmdb'
require 'image'

local config = require 'Config'
local data = require 'Data'
local TrainDB = data.TrainDB
local ValDB = data.ValDB


function BenchmarkDB(DB, Name, num, InputSize, visualize)
  local Num = num or 128
  config.InputSize = InputSize or {3, 224, 224}
  print('\n\n===> ', Name ..  ' DB Benchmark, ' ..  Num .. ' Items')
  local DataSetSize = DB:size()
  print('DB Size: ' .. DataSetSize)
  print('Sample Size: ', config.InputSize)

  local x = torch.FloatTensor(Num ,unpack(config.InputSize))
  local y = torch.IntTensor(Num)


  print('\n==> Synchronous timing')
  local t, key

  key = Key(math.random(DataSetSize-Num))
  t=torch.tic()
  DB:CacheSeq(key,Num ,x,y)
  t=torch.tic()-t
  print('1) Sequential Time: ' .. string.format('%04f',t/Num) .. ' Per sample')
  if visualize then image.display(x) end

  randKeys = Keys(torch.randperm(DataSetSize):narrow(1,1,Num))
  t=torch.tic()
  DB:CacheRand(randKeys,x,y)
  t=torch.tic()-t
  print('2) Random Access Time: ' .. string.format('%04f',t/Num) .. ' Per sample')
  if visualize then image.display(x) end

  print('\n==> Asynchronous timing')
  DB:Threads()
  key = Key(math.random(DataSetSize-Num))
  t=torch.tic()
  DB:AsyncCacheSeq(key,Num ,x,y)
  print('1) Async Sequential Spawn Time: ' .. string.format('%04f',(torch.tic()-t)/Num) .. ' Per sample')
  DB:Synchronize()
  t=torch.tic()-t
  print('   Async Sequential Complete Time: ' .. string.format('%04f',t/Num) .. ' Per sample')
  if visualize then image.display(x) end

  randKeys = Keys(torch.randperm(DataSetSize):narrow(1,1,Num))
  t=torch.tic()
  DB:AsyncCacheRand(randKeys,x,y)
  print('2) Async Random Spawn Time: ' .. string.format('%04f',(torch.tic()-t)/Num) .. ' Per sample')
  DB:Synchronize()
  t=torch.tic()-t
  print('   Async Random Complete Time: ' .. string.format('%04f',t/Num) .. ' Per sample')
  if visualize then image.display(x) end
end

BenchmarkDB(TrainDB, 'Training')
BenchmarkDB(ValDB, 'Validation')
