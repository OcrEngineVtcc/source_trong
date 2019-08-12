import os
import cv2
import shutil
import random
import natsort
import argparse
import numpy as np
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

class DataTool(object):
	def __init__(self, data_folder_path, image_extension):
		super(DataTool, self).__init__()
		self.data_folder_path = Path(data_folder_path)
		self.image_extension = image_extension

	def scan(self):
		raise NotImplementedError

class CropTextline(DataTool):
	def __init__(self, data_folder_path, image_extension):
		super(CropTextline, self).__init__(data_folder_path, image_extension)
		
	def _crop_textline(self, image_path, bbox_path, output_folder_path):
		def tag_type(tag):
			return tag.split('}')[-1]

		image_path, output_folder_path = Path(image_path), Path(output_folder_path)
		image_name = image_path.stem
		image = cv2.imread(image_path.__str__())
		root = ET.parse(bbox_path).getroot()

		for ele in root.iter():
			if tag_type(ele.tag) == 'TextLine':
				id = ele.attrib['id']
				text = ele[1][0].text if ele[1][0].text is not None else ''
				coords = ele[0].attrib['points']
				coords = coords.split()
				coords = np.asarray([list(map(lambda num : int(num), coord.split(','))) for coord in coords], dtype=int)

				width_min, height_min = coords.min(axis=0)
				width_max, height_max = coords.max(axis=0)
				textline_image = image[height_min:height_max, width_min:width_max]

				cv2.imwrite(output_folder_path.joinpath('{}_{}.tif'.format(image_name, id)).__str__(), textline_image)
				with output_folder_path.joinpath('{}_{}.txt'.format(image_name, id)).open(mode='w') as f:
					f.write(text.strip())

	def scan(self):
		image_bbox_folder_path = self.data_folder_path.joinpath('image_bbox')
		image_bbox_folder_path.mkdir()

		images = natsort.natsorted(self.data_folder_path.glob('*{}'.format(self.image_extension)), key=lambda x : x.name)
		bboxes = natsort.natsorted(self.data_folder_path.glob('*_bbox.xml'), key=lambda x : x.name)
		for image, bbox in zip(images, bboxes):
			self._crop_textline(image.__str__(), bbox.__str__(), self.data_folder_path.__str__())
			shutil.move(image.__str__(), image_bbox_folder_path.__str__())
			shutil.move(bbox.__str__(), image_bbox_folder_path.__str__())
		
class GenerateBox(DataTool):
	def __init__(self, data_folder_path, image_extension):
		super(GenerateBox, self).__init__(data_folder_path, image_extension)
		
	def _generate_box(self, textline_image_path, label_path):
		textline_image_path, label_path = Path(textline_image_path), Path(label_path)
		box_path = label_path.with_suffix('.box')
		height, width = cv2.imread(textline_image_path.__str__(), cv2.IMREAD_GRAYSCALE).shape

		with label_path.open(mode='r') as label_file:
				label_text = label_file.readline().strip()
				if label_text == '':
					invalid_folder_path = self.data_folder_path.joinpath('invalid')
					invalid_folder_path.mkdir(exist_ok=True)
					shutil.move(textline_image_path.__str__(), invalid_folder_path.__str__())
					shutil.move(label_path.__str__(), invalid_folder_path.__str__())
				else:
					with box_path.open(mode='w') as box_file:
						for i in range(len(label_text) - 1):
							if unicodedata.combining(label_text[i + 1]):
								box_file.write('{} {} {} {} {} {}\n'.format('{}{}'.format(label_text[i], label_text[i + 1]), 0, 0, width, height, 0))
							elif not unicodedata.combining(label_text[i]):
								box_file.write('{} {} {} {} {} {}\n'.format(label_text[i], 0, 0, width, height, 0))
						box_file.write('{} {} {} {} {} {}\n'.format(label_text[-1], 0, 0, width, height, 0))
						box_file.write('{} {} {} {} {} {}\n'.format('\t', 0, 0, width, height, 0))

	def scan(self):
		textline_image_paths = natsort.natsorted(self.data_folder_path.glob('*{}'.format(self.image_extension)), key=lambda x : x.name)
		label_paths = natsort.natsorted(self.data_folder_path.glob('*.txt'), key=lambda x : x.name)
		for textline_image_path, label_path in zip(textline_image_paths, label_paths):
			self._generate_box(textline_image_path.__str__(), label_path.__str__())

class GenerateLSTMF(DataTool):
	def __init__(self, data_folder_path, image_extension):
		super(GenerateLSTMF, self).__init__(data_folder_path, image_extension)
		
	def _generate_lstmf(self, image_path):
		cmd = 'tesseract {} {} --psm 7 lstm.train'.format(image_path, image_path[:image_path.rfind('.')])
		os.system(cmd)

	def scan(self):
		textline_label_folder_path = self.data_folder_path.joinpath('textline_label')
		textline_label_folder_path.mkdir()
		image_paths = natsort.natsorted(self.data_folder_path.glob('*{}'.format(self.image_extension)), key=lambda x : x.name)
		for image_path in image_paths:
			self._generate_lstmf(image_path.__str__())
			shutil.move(image_path.__str__(), textline_label_folder_path.__str__())
			shutil.move(image_path.with_suffix('.box').__str__(), textline_label_folder_path.__str__())
			print('{} done.'.format(image_path.name))

class DivideDataset(DataTool):
	def __init__(self, data_folder_path, train_ratio, valid_ratio):
		super(DivideDataset, self).__init__(data_folder_path, None)
		self.dataset_path = self.data_folder_path.joinpath('dataset')
		self.train_ratio = train_ratio
		self.valid_ratio = valid_ratio
	
	def _divide(self):
		train_list_path = self.data_folder_path.joinpath('training.list')
		valid_list_path = self.data_folder_path.joinpath('validation.list')
		test_list_path = self.data_folder_path.joinpath('test.list')
		lstmf_paths = list(self.dataset_path.glob('*.lstmf'))
		random.shuffle(lstmf_paths)
		train_lstmf_paths = lstmf_paths[:int(self.train_ratio * len(lstmf_paths))]
		valid_lstmf_paths = lstmf_paths[int(self.train_ratio * len(lstmf_paths)):int((self.train_ratio + self.valid_ratio) * len(lstmf_paths))]
		test_lstmf_paths = lstmf_paths[int((self.train_ratio + self.valid_ratio) * len(lstmf_paths)):]

		with train_list_path.open(mode='w') as f:
			for path in train_lstmf_paths:
				f.write('{}\n'.format(path.__str__()))
		
		with valid_list_path.open(mode='w') as f:
			for path in valid_lstmf_paths:
				f.write('{}\n'.format(path.__str__()))

		with test_list_path.open(mode='w') as f:
			for path in test_lstmf_paths:
				f.write('{}\n'.format(path.__str__()))

	def scan(self):
		self._divide()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', type=str, required=True, help='path to folder containing images and bbox files')
	parser.add_argument('--image-extension', type=str, default='.jpg', help='image extension')
	parser.add_argument('--train-ratio', type=float, default=0.64, help='training ratio')
	parser.add_argument('--valid-ratio', type=float, default=0.16, help='validation ratio')
	parser.add_argument('--fine-turning', action='store_true', default=False, help='fine-turning or not')
	parser.add_argument('--train-from-scratch', action='store_true', default=False, help='train from scratch or not')

	args = parser.parse_args()

	data_folder_path = Path(args.data_path)

	if args.fine_turning:
		os.system('wget -P {} https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/master/vie.traineddata'.format(data_folder_path.__str__()))
		os.system('combine_tessdata -e {} {}'.format(data_folder_path.joinpath('vie.traineddata').__str__(), data_folder_path.joinpath('vie.lstm').__str__()))
	elif args.train_from_scratch:
		all_textline_path = data_folder_path.joinpath('all_textline.txt')
		os.system('wget -P {} https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/master/Latin.unicharset'.format(data_folder_path.__str__()))
		os.system('wget -P {} https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/master/radical-stroke.txt'.format(data_folder_path.__str__()))
		os.system('wget -P {} https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/master/vie/vie.unicharset'.format(data_folder_path.__str__()))
		os.system('cat {} > {}'.format(data_folder_path.joinpath('textline_label').__str__(), all_textline_path.__str__()))
		os.system('combine_lang_model --input_unicharset {} --script_dir {} --output_dir {} --lang vie'.format(data_folder_path.joinpath('vie.unicharset').__str__(), data_folder_path.__str__(), data_folder_path.__str__()))
		all_textline_path.unlink()
		data_folder_path.joinpath('Latin.unicharset').unlink()
		data_folder_path.joinpath('radical-stroke.txt').unlink()
		data_folder_path.joinpath('vie.unicharset').unlink()
		shutil.copy(data_folder_path.joinpath('vie', 'vie.traineddata').__str__(), data_folder_path.__str__())
		shutil.rmtree(data_folder_path.joinpath('vie').__str__())
	else:
		crop_textline = CropTextline(data_folder_path.__str__(), args.image_extension)
		generate_box = GenerateBox(data_folder_path.__str__(), '.tif')
		generate_lstmf = GenerateLSTMF(data_folder_path.__str__(), '.tif')
		divide_dataset = DivideDataset(data_folder_path.__str__(), float(args.train_ratio), float(args.valid_ratio))

		crop_textline.scan()
		generate_box.scan()
		generate_lstmf.scan()

		dataset_path, image_label_box_path = data_folder_path.joinpath('dataset'), data_folder_path.joinpath('image_label_path')
		checkpoint_folder_path, model_folder_path = data_folder_path.joinpath('checkpoints'), data_folder_path.joinpath('model_output')

		dataset_path.mkdir()
		checkpoint_folder_path.mkdir()
		model_folder_path.mkdir()

		box_paths, lstmf_paths = data_folder_path.glob('*.txt'), data_folder_path.glob('*.lstmf')

		for box_path in box_paths:
			box_path.unlink()

		for lstmf_path in lstmf_paths:
			shutil.move(lstmf_path.__str__(), dataset_path.__str__())

		divide_dataset.scan()