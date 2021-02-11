python AdversarialDetection.py --dataset=mnist                                 ^
                               --data_path=E:\Projects\Dataset\mnist\processed ^
                               --batch_size=1                                  ^
                               --mode=adversary                                ^
                               --epsilon=%1                                    ^
                               --cuda=True                                     ^
                               --cnn=E:\Projects\CPSC-597\AdversarialDetection\State\cnn.model