require 'nn'

local function ConvBN(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    local module = nn.Sequential()
    module:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
    module:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3,nil,true))
    module:add(nn.ReLU(true))
    return module
end

local function Pooling(poolType, ...)
    local pooler
    if poolType == 'max' then
        pooler = nn.SpatialMaxPooling(unpack({...}))

    else
        pooler = nn.SpatialAveragePooling(unpack({...}))

    end
    return pooler
end

local function Tower(tbl)
    local module = nn.Sequential()
    for i=1, #tbl do
        module:add(tbl[i])
    end
    return module
end

local function Inception(tbl)
    local module = nn.Concat(2)
    for i=1, #tbl do
        module:add(tbl[i])
    end
    return module
end


local model = nn.Sequential()
model:add(ConvBN(3,32,3,3,2,2,0,0))
model:add(ConvBN(32,32,3,3,1,1,0,0))
model:add(ConvBN(32,64,3,3,1,1,1,1))
model:add(Pooling('max',3,3,2,2,0,0))
model:add(ConvBN(64,80,1,1,1,1,0,0))
model:add(ConvBN(80,192,3,3,1,1,0,0))
model:add(Pooling('max',3,3,2,2,0,0))

model:add(Inception{
    Tower{
        ConvBN(192,64,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(192,48,1,1,1,1,0,0),
        ConvBN(48,64,5,5,1,1,2,2)
    },
    Tower{
        ConvBN(192,64,1,1,1,1,0,0),
        ConvBN(64,96,3,3,1,1,1,1),
        ConvBN(96,96,3,3,1,1,1,1)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(192,32,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(256,64,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(256,48,1,1,1,1,0,0),
        ConvBN(48,64,5,5,1,1,2,2)
    },
    Tower{
        ConvBN(256,64,1,1,1,1,0,0),
        ConvBN(64,96,3,3,1,1,1,1),
        ConvBN(96,96,3,3,1,1,1,1)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(256,64,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(288,64,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(288,48,1,1,1,1,0,0),
        ConvBN(48,64,5,5,1,1,2,2),
    },
    Tower{
        ConvBN(288,64,1,1,1,1,0,0),
        ConvBN(64,96,3,3,1,1,1,1),
        ConvBN(96,96,3,3,1,1,1,1)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(288,64,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(288,384,3,3,2,2,0,0)
    },
    Tower{
        ConvBN(288,64,1,1,1,1,0,0),
        ConvBN(64,96,3,3,1,1,1,1),
        ConvBN(96,96,3,3,2,2,0,0)
    },
    Tower{
        Pooling('max',3,3,2,2,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(768,192,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(768,128,1,1,1,1,0,0),
        ConvBN(128,128,7,1,1,1,3,0),
        ConvBN(128,192,1,7,1,1,0,3)
    },
    Tower{
        ConvBN(768,128,1,1,1,1,0,0),
        ConvBN(128,128,1,7,1,1,0,3),
        ConvBN(128,128,7,1,1,1,3,0),
        ConvBN(128,128,1,7,1,1,0,3),
        ConvBN(128,192,7,1,1,1,3,0)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(768,192,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(768,192,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(768,160,1,1,1,1,0,0),
        ConvBN(160,160,7,1,1,1,3,0),
        ConvBN(160,192,1,7,1,1,0,3)
    },
    Tower{
        ConvBN(768,160,1,1,1,1,0,0),
        ConvBN(160,160,1,7,1,1,0,3),
        ConvBN(160,160,7,1,1,1,3,0),
        ConvBN(160,160,1,7,1,1,0,3),
        ConvBN(160,192,7,1,1,1,3,0)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(768,192,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(768,192,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(768,160,1,1,1,1,0,0),
        ConvBN(160,160,7,1,1,1,3,0),
        ConvBN(160,192,1,7,1,1,0,3)
    },
    Tower{
        ConvBN(768,160,1,1,1,1,0,0),
        ConvBN(160,160,1,7,1,1,0,3),
        ConvBN(160,160,7,1,1,1,3,0),
        ConvBN(160,160,1,7,1,1,0,3),
        ConvBN(160,192,7,1,1,1,3,0)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(768,192,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(768,192,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(768,192,1,1,1,1,0,0),
        ConvBN(192,192,7,1,1,1,3,0),
        ConvBN(192,192,1,7,1,1,0,3),
    },
    Tower{
        ConvBN(768,192,1,1,1,1,0,0),
        ConvBN(192,192,1,7,1,1,0,3),
        ConvBN(192,192,7,1,1,1,3,0),
        ConvBN(192,192,1,7,1,1,0,3),
        ConvBN(192,192,7,1,1,1,3,0)
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(768,192,1,1,1,1,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(768,192,1,1,1,1,0,0),
        ConvBN(192,320,3,3,2,2,0,0)
    },
    Tower{
        ConvBN(768,192,1,1,1,1,0,0),
        ConvBN(192,192,7,1,1,1,3,0),
        ConvBN(192,192,1,7,1,1,0,3),
        ConvBN(192,192,3,3,2,2,0,0)
    },
    Tower{
        Pooling('max',3,3,2,2,0,0)
    }
})

model:add(Inception{
    Tower{
        ConvBN(1280,320,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(1280,384,1,1,1,1,0,0),
        Inception{
            Tower{
                ConvBN(384,384,3,1,1,1,1,0)
            },
            Tower{
                ConvBN(384,384,1,3,1,1,0,1)
            }
        }
    },
    Tower{
        ConvBN(1280,448,1,1,1,1,0,0),
        ConvBN(448,384,3,3,1,1,1,1),
        Inception{
            Tower{
                ConvBN(384,384,3,1,1,1,1,0)
            },
            Tower{
                ConvBN(384,384,1,3,1,1,0,1)
            }
        }
    },
    Tower{
        Pooling('average',3,3,1,1,1,1),
        ConvBN(1280,192,1,1,1,1,0,0)
    }
})
model:add(Inception{
    Tower{
        ConvBN(2048,320,1,1,1,1,0,0)
    },
    Tower{
        ConvBN(2048,384,1,1,1,1,0,0),
        Inception{
            Tower{
                ConvBN(384,384,3,1,1,1,1,0)
            },
            Tower{
                ConvBN(384,384,1,3,1,1,0,1)
            }
        }
    },
    Tower{
        ConvBN(2048,448,1,1,1,1,0,0),
        ConvBN(448,384,3,3,1,1,1,1),
        Inception{
            Tower{
                ConvBN(384,384,3,1,1,1,1,0)
            },
            Tower{
                ConvBN(384,384,1,3,1,1,0,1)
            }
        }
    },
    Tower{
        Pooling('max',3,3,1,1,1,1),
        ConvBN(2048,192,1,1,1,1,0,0)
    }
})


model:add(Pooling('average',8,8,1,1,0,0))
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Dropout(0.2))
model:add(nn.Linear(2048,1000))
model:add(nn.LogSoftMax())


return {
  model = model,
  rescaleImgSize = 384,
  InputSize = {3, 299, 299}
}
