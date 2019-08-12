import cv2
import csv
import json
import numpy as np
import pytesseract
from io import StringIO
from word_box import box, _space_info, _trunc

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
			left_trunc, right_trunc = _trunc(textline_image, axis=0)
			space_info = _space_info(textline_image[:, left_trunc:right_trunc], axis=0)
			if len(space_info) > 1:
				space_info = np.asarray(space_info)
				space_info = space_info[:, 1] - space_info[:, 0]
				threshold_h = 2 * space_info.mean() * space_info.max() / (space_info.mean() + space_info.max())
			else:
				threshold_h = float('inf')

			textline['text'] = pytesseract.image_to_string(textline_image, lang=lang, config='--psm 7 --oem 1')
			
			textline_tsv = pytesseract.image_to_data(textline_image, lang=lang, config='--psm 7 --oem 1', output_type=pytesseract.Output.DICT)
			words = []
			for i in range(len(textline_tsv['level'])):
				if textline_tsv['conf'][i] != '-1':
					left = left_trunc + x_start + textline_tsv['left'][i]
					top = y_start + textline_tsv['top'][i]
					width, height = textline_tsv['width'][i], textline_tsv['height'][i]
					word_image = image[top:top + height, left:left + width]
					left_box, top_box, right_box, bottom_box = box(word_image, threshold_h) if min(width, height) > 3 else (0, 0, width, height)
					# left_box, top_box, right_box, bottom_box = (0, 0, width, height)

					words.append({
							'word': textline_tsv['text'][i],
							'word_coord': [
											{'x': left + left_box, 'y': top + top_box}, 
											{'x': left + right_box, 'y': top + top_box}, 
											{'x': left + right_box, 'y': top + bottom_box}, 
											{'x': left + left_box, 'y': top + bottom_box}
										]
						})
			textline['words'] = words
		with open(json_path.replace('.json', '_ocr.json'), mode='w') as f:
			json.dump(obj, f)