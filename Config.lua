local ImageNetClasses = torch.load('./ImageNetClasses')
for i=1001,#ImageNetClasses.ClassName do
    ImageNetClasses.ClassName[i] = nil
end

function Key(num)
    return string.format('%07d',num)
end


return
{
    TRAINING_PATH = '/home/ehoffer/Datasets/ImageNet/train/',
    VALIDATION_PATH = '/home/ehoffer/Datasets/ImageNet/validation/',
    VALIDATION_DIR = '/home/ehoffer/Datasets/ImageNet/LMDB_Compressed/validation/',
    TRAINING_DIR = '/home/ehoffer/Datasets/ImageNet/LMDB_Compressed/train/',
    ImageSize = 256,
    SampleSize = {3,224,224},
    ValidationLabels = torch.load('./ValidationLabels'),
    ImageNetClasses = ImageNetClasses,
    DataMean = 118.380948,
    DataStd = 61.896913,
    Compressed = true,
    Key = Key
}



