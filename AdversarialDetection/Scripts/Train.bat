@echo off
REM echo Train Convolutional Neural Network
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=train                                                               ^
REM    --epochs=100                                                               ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                

REM echo Train Reconstructor
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=recon                                                               ^
REM    --epochs=50                                                                ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model

REM echo Train Auxiliary Classifier (Classifier for each Convolutional Layer)
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=classifier                                                          ^
REM    --epochs=10                                                                ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10classifier.model

REM echo Train Convolutional Neural Network on combined Original and Adversarial Images 
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=atrain                                                              ^
REM    --epsilon=%1                                                               ^
REM    --epochs=100                                                               ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --acnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10acnn.model

echo Train Meta Linear Neural Network to detect Adversarial Images based on CNN and Auxiliary Classifier
python AdversarialDetection.py                                                            ^
   --dataset=cifar10                                                                      ^
   --data_path=E:\Projects\Dataset\cifar10                                                ^
   --batch_size=64                                                                        ^
   --mode=metatrain                                                                       ^
   --epsilon=%1                                                                           ^
   --epochs=2000                                                                          ^
   --cuda=True                                                                            ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                 ^
   --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10classifier.model   ^
   --metann=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10meta.model
   
echo Train Meta CNN on Original Image and Reconstructed Images
python AdversarialDetection.py                                                            ^
   --dataset=cifar10                                                                      ^
   --data_path=E:\Projects\Dataset\cifar10                                                ^
   --batch_size=64                                                                        ^
   --mode=metacnn                                                                         ^
   --epsilon=%1                                                                           ^
   --epochs=100                                                                           ^
   --cuda=True                                                                            ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                 ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model             ^
   --metacnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10metacnn.model

