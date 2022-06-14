from torch.utils.data import Dataset

class XYDataset(Dataset):
	def __init__(self, x_ten, y_ten):
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

if __name__ == '__main__':
	X_IND = 21
	Y_IND = 1
	dir = ['ptbxl', 'ptbxl2', 'georgia', 'china', 'cpsc', 'ptb', 'stpet']
	val = [200, 200, 200, 149, 251, 0, 0]
	val21 = [400, 400, 400, 0, 400, 0, 0]
	x_list = []
	y_list = []
	for s in dir:
		cnt1 = 0
		cnt21 = 0
  		pathD = s + '.pt'
  		pathL = s + 'labels.pt'
		tensor = torch.load(pathD)
  		tlabels = torch.load(pathL)
  		for i in range(len(tensor)):
    		if int(tlabels[i]) == 1:
      			cnt1 += 1	
      		if int(tlabels[i]) == 21:
      			cnt21 += 1
      		if cnt21 <= 1600:
      			x_list.append(tensor[i])
      		if cnt1 <= 1000:
      			y_list.append(tensor[i])