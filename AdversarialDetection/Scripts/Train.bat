@echo off
python AdversarialDetection.py                                                ^
   --dataset=cifar10                                                          ^
   --data_path=E:\Projects\Dataset\cifar10                                    ^
   --batch_size=64                                                            ^
   --mode=train                                                               ^
   --epochs=100                                                               ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                

python AdversarialDetection.py                                                ^
   --dataset=cifar10                                                          ^
   --data_path=E:\Projects\Dataset\cifar10                                    ^
   --batch_size=64                                                            ^
   --mode=recon                                                               ^
   --epochs=50                                                                ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model

python AdversarialDetection.py                                                ^
   --dataset=cifar10                                                          ^
   --data_path=E:\Projects\Dataset\cifar10                                    ^
   --batch_size=64                                                            ^
   --mode=classifier                                                          ^
   --epochs=1                                                                 ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
   --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10classifier.model
                               
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