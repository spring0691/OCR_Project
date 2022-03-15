1. pths 폴더를 만들고 vgg16 전이학습 모델(https://drive.google.com/open?id=1HgDuFGd2q77Z6DcUlDEfBZgxeJv4tald)과 east-vgg16 가중치(https://drive.google.com/open?id=1AFABkJgr5VtxWnmBU3XcfLJvpZkC2TAg)를 다운받는다.
   
2. dataset - model - loss - train - eval - detect 순으로 파일을 확인하자

3. visual studio를 설치, C++ compiler에 동의 후 lanms-neo를 pip install

4. model.py에 import os랑 path = os.path.dirname(os.path.realpath(__file__))와 
   69번째 줄 torch.load(f'{path}/pths/vgg16_bn-6c64b313.pth') 추가하여 수정했습니다. - 예람

5. train.py 현재 없는데 업데이트 새로하고 5번 주석 지워주세요.
