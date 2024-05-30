import os

imgdirlist = ['訓練集', '測試集', '驗證集']
output = ['train.txt', 'test.txt', 'valid.txt']

for i in range(3):
	imgpath = os.path.join('dataset', imgdirlist[i])
	imglist = os.listdir(imgpath)
	with open(os.path.join('data', output[i]), 'w') as f:
		for name in imglist:
			f.write(os.path.join(imgpath, name)+'\n')