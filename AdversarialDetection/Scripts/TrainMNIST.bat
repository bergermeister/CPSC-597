@echo off
echo Train Convolutional Neural Network
python AdversarialDetection.py                                                ^
   --dataset=mnist                                                            ^
   --data_path=E:\Projects\Dataset\mnist                                      ^
   --cuda=True                                                                ^
   --batch_size=64                                                            ^
   --mode=cnn                                                                 ^
   --epochs=100                                                               ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model                

echo Train Reconstructor
python AdversarialDetection.py                                                ^
   --dataset=mnist                                                            ^
   --data_path=E:\Projects\Dataset\mnist                                      ^
   --cuda=True                                                                ^
   --batch_size=64                                                            ^
   --mode=recon                                                               ^
   --epochs=50                                                                ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model     ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\mnistrecon.model
  
echo Train Meta CNN on Original Image and Reconstructed Images
python AdversarialDetection.py                                                            ^
   --dataset=mnist                                                                        ^
   --data_path=E:\Projects\Dataset\mnist                                                  ^
   --cuda=True                                                                            ^
   --batch_size=64                                                                        ^
   --mode=detect                                                                         ^
   --epsilon=%1                                                                           ^
   --epochs=100                                                                           ^
   --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\mnistcnn.model                 ^
   --recon=E:\Projects\CPSC-597\AdversarialDetection\State\mnistrecon.model             ^
   --detect=E:\Projects\CPSC-597\AdversarialDetection\State\mnistdetect.model

