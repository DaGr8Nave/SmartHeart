import torch
from torch.utils.data import Dataset

class XYDataset(Dataset):
	def __init__(self, X_IND, Y_IND):
		dir = ['ptbxl', 'ptbxl2', 'georgia', 'china', 'cpsc', 'ptb', 'stpet']
		val = [200, 200, 200, 149, 251, 0, 0]
		val21 = [400, 400, 400, 0, 400, 0, 0]
		root = '../../../../input/cinc2020bandpassf/'
		j = 0
		x_ten = []
		y_ten = []
		for s in dir:
			cnt1 = 0
			cnt21 = 0
			pathD = root + s + '.pt'
			pathL = root + s + 'labels.pt'
			
			if s == 'ptbxl2':
				pathL = root + 'ptbxllabels2.pt'
			tensor = torch.load(pathD)
			tlabels = torch.load(pathL)
			
			for i in range(len(tensor)):
				if int(tlabels[i]) == 1:
					cnt1 += 1	
				if int(tlabels[i]) == 21:
					cnt21 += 1
				v = tensor[i].clone()
				if cnt21 <= val21[j]: 
					x_ten.append(v)
				if cnt1 <= val[j]:
					y_ten.append(v)
				if cnt1 > val[j] and cnt21 > val21[j]:
					break
				del v
			del tensor
			del tlabels
			
			j+=1
		torch.cuda.empty_cache()
		x_ten = torch.vstack((x_ten))
		y_ten = torch.vstack((y_ten))
		self.x_ten = x_ten
		self.y_ten = y_ten
	
		self.length_dataset = max(len(self.x_ten), len(self.y_ten))
		self.x_len = len(self.x_ten)
		self.y_len = len(self.y_ten)
	
	def __len__(self):
		return self.length_dataset

	def __getitem__(self, index):
		x = self.x_ten[index % self.x_len]
		y = self.y_ten[index % self.y_len]

		return (x,y)
