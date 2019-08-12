import cv2
import csv
import json
import numpy as np
import pytesseract
from io import StringIO

class OCR(object):
	@staticmethod
	def textline(json_path, lang='vie'):
		with open(json_path, mode='r') as f:
			obj = json.load(f)

		image = cv2.imread(obj['image_dir'])

		for textline in obj['text_lines']:
			start = min(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			stop = max(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			x_start, y_start = start['x'], start['y']
			x_stop, y_stop = stop['x'], stop['y']
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
					words.append({
							'word': row[-1],
							'word_coord': [
											{'x': left, 'y': top}, 
											{'x': left + width, 'y': top}, 
											{'x': left + width, 'y': top + height}, 
											{'x': left, 'y': top + height}
										]
						})
			textline['words'] = words
		with open(json_path.replace('.json', '_ocr.json'), mode='w') as f:
			json.dump(obj, f)