from torch.utils.data import DataLoader
from PIL import Image 
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import  CIFAR10, CelebA
from torchvision import transforms
import os

import torch
import numpy as np

from pydicom import dcmread
 
class LDCTLoader():
    def __init__(self, root, batch_size, shuffle=False):

        self.dataset = LDCTDataset(root)

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle, sampler=DistributedSampler(self.dataset))

class LDCTDataset(Dataset):

	"""
	Dataset for the LDCT and Projection data. This assumes that for each series, there will be a folder for the full dose images and a folder for the low dose images.
	
	Attributes:
		root (str): path to the data of the series.
		item_as_series (bool): whether the item returned by __getitem__() should be one of the ~1,000 series or one of the ~1,000,000 individuals CT slices.
		transform (callable): transforms to the applied.
		ld_path (str): the path which when put together with a specific series path will point to that series's low dose images (i.e., os.path.join(self.series_list[i], self.ld_path) will point to the series of low dose images).
		fd_path (str): same as ld_path but for full dose images.
		series_list (list): list of the paths of each series within the root folder.
		ld_image_list (list): list of the paths to each individual low dose image.
		fd_image_list (list): list of the paths to each individual full dose image.
	"""
	
	#TODO: replace ld_path and fd_path with what they will actually be!
	def __init__(self, root: str, item_as_series: bool = False, transform = None, ld_path: str = 'LDCT', fd_path: str = 'FDCT'):
	
		"""
		Initiates the dataset.
		
		Args:
			root (str): the path to the data of the series.
			item_as_series (bool, optional): whether or not to return each item as a full series or an individual slice.
			transform (callable, optional): transform to be applied on a sample.
			ld_path (str, optional): the extension to the low dose path when combined with a series path.
			fd_path (str, optional): the same but with the full dose path.
		"""
		
		self.root = root
		self.item_as_series = item_as_series
		self.transform = transform
		
		self.ld_path = ld_path
		self.fd_path = fd_path
		
		self.series_list = [ f.path for f in os.scandir(self.root) if f.is_dir() ]
	
		self.assert_same_length()
		
		if item_as_series == False:
			self.ld_image_list, self.fd_image_list = self.get_list_of_image_paths()
			assert len(self.ld_image_list) == len(self.fd_image_list), f'Your low dose and full dose image lists do not have the same length! You did pass the assertion earlier, so your get_list_of_image_paths must not be working as intended.'
	
	def assert_same_length(self):
		
		"""
		Ensures that for each series, the low dose and full dose have the same amount of slices. Raises an error otherwise
		"""
		
		for series in self.series_list:
			ld_set = os.listdir( os.path.join(series, self.ld_path) )
			fd_set = os.listdir( os.path.join(series, self.fd_path) )
			
			if len(ld_set) != len(fd_set):
				raise RuntimeException(f'The length of the low dose and full dose CT images are not the same for the series {series}. The low dose has {len(ld_set)} and the full dose has {len(fd_set)} slices. Please remedy this before proceeding')
		
		print('Passed the same length test')
	
	def get_list_of_image_paths(self):
	
		"""
		Gets a list of images for more efficient item getting when the image is treated as the item.
		
		Returns:
			tuple: two lists of strings which contain the low dose and full dose paths, respectively.
		"""
		
		ld_images = []
		fd_images = []
		for series in self.series_list:
			ld_set = [ dcm.path for dcm in os.scandir(os.path.join(series, self.ld_path)) if dcm.is_file() ]
			fd_set = [ dcm.path for dcm in os.scandir(os.path.join(series, self.fd_path)) if dcm.is_file() ]
			
			ld_images.extend(ld_set)
			fd_images.extend(fd_set)
			
		return (ld_images, fd_images)
		
	def __len__(self) -> int:
	
		"""
		Gets the length of the dataset.
		
		Returns:
			int: ___ or ___ depending on if a series is treated as the item or if the image is.
		"""
		
		if self.item_as_series:
			return len(self.series_list)
		else:
			return len(self.ld_image_list)
		
	def __getitem__(self, index):
	
		"""
		Gets the item at the specified index. The item depends on if the series is treated as the item or if the image is.
		
		Args:
			index (int): the index of the item which is being accessed.
		
		Returns:
			tuple: two tensors which are both of shape [N, 1, 512, 512] if the series is treated as the item, or [64, 1, 64, 64] if the image is (the image is split into 64 patches). N represents the number of slices in a series. The first tensor will always be the low dose, and the second the full dose.
		"""
		
		if self.item_as_series:
			return self.get_series_item(index)
		else:
			x = self.get_image_item(index)
			#x = x.tensor_split(8, 1)
			#ls = []

			#for patch in x:
			#	ls.extend(patch.tensor_split(8, 2))
			
			#x = torch.stack(ls)
			if self.transform:
				x = self.transform(x)
			return x
			
	def get_series_item(self, index):
	
		"""
		Gets a series at the specified index.
		
		Args:
			index (int): the index of the series being accessed.
			
		Returns:
			tuple: tuple of the noisy series and the non-noisy series with dimensions [N, 1, 512, 512] where N is the number of slices in the specified series. The tensor has a minimum value of 0 and a maximum of 255, and its data type of uint8.
		"""
		
		ld_series = torch.stack([ self.get_normalized_CT_slice(dcm.path) for dcm in os.scandir(os.path.join(self.series_list[index], self.ld_path)) if dcm.is_file()])
		fd_series = torch.stack([ self.get_normalized_CT_slice(dcm.path) for dcm in os.scandir(os.path.join(self.series_list[index], self.fd_path)) if dcm.is_file()])
		
		if self.transform:
			ld_series = self.transform(ld_series)
			fd_series = self.transform(fd_series)
			
		return (ld_series, fd_series)
		
	def get_image_item(self, index):
	
		"""
		Gets a specific DICOM slice as a tensor of shape [1, 512, 512]
		
		Args:
			index (int): the index of the slice being accessed
		
		Returns:
			tuple: a tensor representation of the CT slice at the specified index for the noisy series and the non-noisy series. The data type is uint8, min value is 0, and max value is 255. The shape of the tensor is [1, 512, 512]
		"""
		
		ld_slice = self.get_normalized_CT_slice(self.ld_image_list[index])
		fd_slice = self.get_normalized_CT_slice(self.fd_image_list[index])
		
		if self.transform:
			ld_slice = self.transform(ld_slice)
			fd_slice = self.transform(fd_slice)
		
		return (ld_slice, fd_slice)
		
	def get_normalized_CT_slice(self, path) -> torch.Tensor:
	
		"""
		Gets a specific DICOM file as a tensor of shape [1, 512, 512]. 
		
		Args:
			path: (str): the path of the tensor
			
		Returns:
			Tensor: tensor representation of the CT slice of type uint8, min value of 0 and max value of 255, and shape [1, 512, 512]
		"""

		dcm = dcmread(path)
		image = dcm.pixel_array
		
		#image = (image - np.min(image)) / np.max(image)
		#image = (image * 255).astype(np.uint8)
		
		image = image.astype(np.float32)
		
		image = torch.from_numpy(image)
		
		image = torch.unsqueeze(image, 0)
		
		return image
		
 
