import cv2
import csv
import json
import pytesseract
from io import StringIO

class OCR(object):
	@staticmethod
	def textline(json_path, lang='vie'):
		with open(json_path, mode='r') as f:
			obj = json.load(f)

		image = cv2.imread(obj['image_dir'])
		image_height, image_width, _ = image.shape

		for textline in obj['text_lines']:
			x, y, w, h = textline['x'], textline['y'], textline['w'], textline['h']
			x_start, y_start, x_stop, y_stop = x, y, x + w, y + h
			textline_image = image[y_start:y_stop, x_start:x_stop]

			textline['text'] = pytesseract.image_to_string(textline_image, lang=lang, config='--psm 7 --oem 1')
			
			textline_tsv = StringIO(pytesseract.image_to_data(textline_image, lang=lang, config='--psm 7 --oem 1'))
			textline_tsv = list(csv.reader(textline_tsv, delimiter='\t'))
			words = []
			for row in textline_tsv[1:]:
				if row[-2] != '-1':
					left = x_start + int(row[-6])
					top = y_start + int(row[-5])
					width, height = int(row[-4]), int(row[-3])
					words.append([row[-1], {'x': left, 'y': top, 'w': width, 'h':height}])
			textline['words'] = words

		with open(json_path, mode='w') as f:
			json.dump(obj, f)