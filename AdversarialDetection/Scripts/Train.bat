@echo off
echo Train Convolutional Neural Network
python AdversarialDetection.py                                                ^
   --dataset=cifar10                                                          ^
   --data_path=E:\Projects\Dataset\cifar10                                    ^
   --cuda=True                                                                ^
   --batch_size=64                                                            ^
   --mode=cnn                                                                 ^
   --epochs=100                                                               ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                

echo Train Reconstructor
python AdversarialDetection.py                                                ^
   --dataset=cifar10                                                          ^
   --data_path=E:\Projects\Dataset\cifar10                                    ^
   --cuda=True                                                                ^
   --batch_size=64                                                            ^
   --mode=recon                                                               ^
   --epochs=50                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model     ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model
  
echo Train Meta CNN on Original Image and Reconstructed Images
python AdversarialDetection.py                                                            ^
   --dataset=cifar10                                                                      ^
   --data_path=E:\Projects\Dataset\cifar10                                                ^
   --cuda=True                                                                            ^
   --batch_size=64                                                                        ^
   --mode=detect                                                                         ^
   --epsilon=%1                                                                           ^
   --epochs=100                                                                           ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10cnn.model                 ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10recon.model             ^
   --detect=E:\Projects\CPSC-597\AdversarialDetection\State\cifar10detect.model

