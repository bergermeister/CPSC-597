@echo off
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=train                                                               ^
REM    --epochs=100                                                               ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                
REM 
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=recon                                                               ^
REM    --epochs=50                                                                ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model
REM 
REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=classifier                                                          ^
REM    --epochs=10                                                                ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10classifier.model
REM
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

REM python AdversarialDetection.py                                                ^
REM    --dataset=cifar10                                                          ^
REM    --data_path=E:\Projects\Dataset\cifar10                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=metatrain                                                           ^
REM    --epsilon=%1                                                               ^
REM    --epochs=2000                                                              ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
REM    --metann=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10meta.model ^
REM    --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10classifier.model

