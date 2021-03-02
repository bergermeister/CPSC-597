import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
import random 
import matplotlib.pyplot as plt

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            print("Processing up to sample=", batchTracker)
            if device== None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Returns the train and val loaders  
def LoadFashionMNISTAsPseudoRGB(batchSize):
    #First transformation, just convert to tensor so we can add in the color channels 
    transformA= transforms.Compose([
        transforms.ToTensor(),
    ])
    #Make the train loader 
    trainLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTrain = len(trainLoader.dataset) 
    sampleIndex = 0
    #This part hard coded for Fashion-MNIST
    xTrain = torch.zeros(numSamplesTrain, 3, 28, 28)
    yTrain = torch.zeros((numSamplesTrain), dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(trainLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTrain[sampleIndex,0] = input[batchIndex]
            xTrain[sampleIndex,1] = input[batchIndex]
            xTrain[sampleIndex,2] = input[batchIndex]
            yTrain[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    #Make the validation loader 
    valLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTest = len(valLoader.dataset) 
    sampleIndex = 0 #reset the sample index to use with the validation loader 
    #This part hard coded for Fashion-MNIST
    xTest = torch.zeros(numSamplesTest, 3, 28, 28)
    yTest = torch.zeros((numSamplesTest),dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(valLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTest[sampleIndex,0] = input[batchIndex]
            xTest[sampleIndex,1] = input[batchIndex]
            xTest[sampleIndex,2] = input[batchIndex]
            yTest[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    transform_train = torch.nn.Sequential(
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform_test = torch.nn.Sequential(
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    trainLoaderFinal = TensorToDataLoader(xTrain, yTrain, transform_train, batchSize, True)
    testLoaderFinal = TensorToDataLoader(xTest, yTest, transform_test, batchSize)
    return trainLoaderFinal, testLoaderFinal

#Show 20 images, 10 in first and row and 10 in second row 
def ShowImages(xFirst, xSecond):
    n = 10  # how many digits we will display
    plt.figure(figsize=(5, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirst[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecond[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation224(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader


