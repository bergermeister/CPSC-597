@echo off
python AdversarialDetection.py                                                ^
   --dataset=mnist                                                            ^
   --data_path=E:\Projects\Dataset\mnist                                      ^
   --batch_size=64                                                            ^
   --mode=train                                                               ^
   --epochs=10                                                                ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model                

python AdversarialDetection.py                                                ^
   --dataset=mnist                                                            ^
   --data_path=E:\Projects\Dataset\mnist                                      ^
   --batch_size=64                                                            ^
   --mode=recon                                                               ^
   --epochs=10                                                                ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model       ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\mnistrecon.model

python AdversarialDetection.py                                                ^
   --dataset=mnist                                                            ^
   --data_path=E:\Projects\Dataset\mnist                                      ^
   --batch_size=64                                                            ^
   --mode=classifier                                                          ^
   --epochs=1                                                                 ^
   --cuda=True                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model       ^
   --classifier=E:\Projects\CPSC-597\AdversarialDetection\State\mnistclassifier.model
                               
REM python AdversarialDetection.py                                                ^
REM    --dataset=mnist                                                          ^
REM    --data_path=E:\Projects\Dataset\mnist                                    ^
REM    --batch_size=64                                                            ^
REM    --mode=atrain                                                              ^
REM    --epsilon=%1                                                               ^
REM    --epochs=100                                                               ^
REM    --cuda=True                                                                ^
REM    --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model     ^
REM    --acnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistacnn.model