import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import os

path = os.path.dirname(os.path.realpath(__file__))		

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']		# 각 레이어의 노드 개수 + 'M'은 MaxPooling 


def make_layers(cfg, batch_norm=False):																# 함수 정의 cfg, batch_norm 매개변수로 지정, batch_norm을 True False 하겠다.
	layers = []																						# 레이어를 저장할 리스트
	in_channels = 3																					# Conv의 채널값. 3은 컬러 1은 흑백
	for v in cfg:																					# cfg의 개수만큼 반복문 실행. -> 18번
		if v == 'M':																				# M일때 Maxpool2d 실행
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)							# cfg의 인자 v가 filters의 개수가 됨.
			if batch_norm:																			# 조건문에 따라서 Batch_norm 실행 + 렐루 또는 렐루만 적용
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v																			# 다음 2번째 레이어부터는 채널값이 곧 이전 레이어의 filters값을 상속받으며 반복실행
	return nn.Sequential(*layers)																	# Sequential에 모든 레이어가 들어간 결과로 return


class VGG(nn.Module):																				# VGG 클래스 선언. nn.Module이 매개변수
	def __init__(self, features):																	# 생성자 함수 선언. 매개변수는 features
		super(VGG, self).__init__()																	# 자신을 생속받고 생성자 호출
		self.features = features																	# features 지정
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))													# avgpool 지정
		self.classifier = nn.Sequential(															# 분류모델 지정
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():														# modules 안의 요소들로 반복 (레이어)
			if isinstance(m, nn.Conv2d):				# isinstance(a,b) a의 속성이 b인지 알아본다. 맞다면 True 틀리다면 False if문은 결국 True를 받아야 실행되므로 m이 Conv2d일때 조건문은 실행된다.
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')	# input tensor가 He초기값으로 U(-bound,bound)의 균일분포를 갖게 해줌 즉 가중치 초기화.
				if m.bias is not None:													# bias가 None이 아닐때 즉 bias가 있을때
					nn.init.constant_(m.bias, 0)										# input tensor를 val값으로 채움 -> m.bias를 0으로 채움
			elif isinstance(m, nn.BatchNorm2d):											# m이 BacthNorm일때
				nn.init.constant_(m.weight, 1)											# weight -> 1로 채움
				nn.init.constant_(m.bias, 0)											# bias -> 0으로 채움
			elif isinstance(m, nn.Linear):												# m이 Linear일때
				nn.init.normal_(m.weight, 0, 0.01)										# input tensor를 N(mean,str^2)의 정규분포에 따라 초기화
				nn.init.constant_(m.bias, 0)											# bias -> 0으로 채움

	def forward(self, x):						# forward 함수 선언 매개변수는 x
		x = self.features(x)					# Functional 방식으로 연결하여 x 반환
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)				# view는 numpy의 reshape와 같은 역할. tensor의 크기를 변환.	 x.size(0) = x의 0차원 사이즈 
		x = self.classifier(x)
		return x


class extractor(nn.Module):																# extractor 클래스 선언
	def __init__(self, pretrained):														# 생성자 함수 선언 초기값 pretrained가 class사용시 필요.
		super(extractor, self).__init__()												# 자신을 상속받고 생성자 재호출
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))								# VGG클래스에 make_layers 함수를 매개변수로 하여 vgg16_bn 16정의.
		if pretrained:																	# pretrained 있기때문에 무조건 실행
			vgg16_bn.load_state_dict(torch.load(f'{path}/pths/vgg16_bn-6c64b313.pth'))	# load_state_dict. torch에서 훈련후 저장한 모델 및 tensor정보등을 역 직렬화된 state_dict를 사용하여
		self.features = vgg16_bn.features	# 모델의 매개변수들을 load해 오는것. state_dict는 간단히 말해 각 체층을 매개변수 Tensor로 매핑한 Python 사전(dict) 객체이다. features값은 vgg16의 features값이 된다
		
	
	def forward(self, x):					# forward 함수 정의
		out = []							# 결과를 담을 list
		for m in self.features:				# self.features에서 반복
			x = m(x)						# x = m(x)로 갱신
			if isinstance(m, nn.MaxPool2d): # m이 Maxpool일때
				out.append(x)				# out에 x 추가
		return out[1:]                      # out에서 앞에부분 제외한 나머지 리턴


class merge(nn.Module):							# merge 클래스 선언
	def __init__(self):							# 생성자 함수 선언
		super(merge, self).__init__()			# 재호출

		self.conv1 = nn.Conv2d(1024, 128, 1)	# 레이어 선언
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		for m in self.modules():				 # 위와 같음
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):						
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)		# Feature의 크기를 변경시킬때 interpolate 사용. size또는 scale_factor로 목표 사이즈 지정. mode는 upsampling 적용방식지정, align_corners는 가장자리 처리 방법 지정
		y = torch.cat((y, x[2]), 1)															# tensor를 concatenate 해주는 함수 cat. 즉 합치겠다
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		y = self.relu7(self.bn7(self.conv7(y)))											
		return y																			# Functional 방식으로 진행하여 y값 반환

class output(nn.Module):						# 함수 선언
	def __init__(self, scope=512):				# 생성자 + 매개변수 scope 지정.
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():				# 상동
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):						# 실행함수 선언 + 매개변수 x
		score = self.sigmoid1(self.conv1(x))					# 점수
		loc   = self.sigmoid2(self.conv2(x)) * self.scope		# location? 아마도 위치값? * scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi	# 각도
		geo   = torch.cat((loc, angle), 1) 						# geo metrics 점수
		return score, geo										# 스코어와 geo값 반환
		
	
class EAST(nn.Module):											# EAST 클래스 선언
	def __init__(self, pretrained=True):						# 생성자 + 매개변수 
		super(EAST, self).__init__()							# 재호출
		self.extractor = extractor(pretrained)					# 위에서 만들어놓은 extractor을 EAST에 내장시킴
		self.merge     = merge()								# merge() 내장
		self.output    = output()								# output 내장
	
	def forward(self, x):										# 실행함수 + 매개변수
		return self.output(self.merge(self.extractor(x)))		# Funtional 방식을 1자로 풀어서 쓴 것. 결국 위에서부터 진행하여 output값 리턴.
		

if __name__ == '__main__':										# 해당 모듈(또는 파일)이 import가 아닌 직접 실행시켰을 경우에만 아래 내용이 실행되게 함
	m = EAST()													# m은 EAST(). 클래스호출
	x = torch.randn(1, 3, 256, 256)								# x 값 지정.
	score, geo = m(x)											# x값 기준으로 EAST() 실행시켜서 score와 geo값 계산.
	print(score.shape)
	print(geo.shape)
