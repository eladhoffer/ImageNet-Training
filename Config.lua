local ImageNetClasses = torch.load('./ImageNetClasses')
for i=1001,#ImageNetClasses.ClassName do
    ImageNetClasses.ClassName[i] = nil
end

return
{
    TRAINING_PATH = '/home/ehoffer/Datasets/ImageNet/train/',
    VALIDATION_PATH = '/home/ehoffer/Datasets/ImageNet/validation/',
    VALIDATION_DIR = './Data/ValidationCache/',
    TRAINING_DIR = './Data/TrainingCache/',
    ImageSize = 256,
    SampleSize = {3,256,256},
    ValidationLabels = torch.load('./ValidationLabels'),
    ImageNetClasses = ImageNetClasses,
    DataMean = 118.380948,
    DataStd = 61.896913
}



