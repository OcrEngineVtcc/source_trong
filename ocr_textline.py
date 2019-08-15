import cv2
import json
import numpy as np
import pytesseract

class OCR(object):
	@staticmethod
	def textline(json_path, lang):
		with open(json_path, mode='r') as f:
			obj = json.load(f)

		image = cv2.imread(obj['image_dir'])

		for i, textline in enumerate(obj['text_lines']):
			start = min(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			stop = max(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			x_start, y_start = start['x'], start['y']
			x_stop, y_stop = stop['x'], stop['y']
			textline_image = image[y_start:y_stop, x_start:x_stop]

			textline['text'] = pytesseract.image_to_string(textline_image, lang=lang, config='--psm 6 --oem 1')
			textline_tsv = pytesseract.image_to_data(textline_image, lang=lang, config='--psm 6 --oem 1', output_type=pytesseract.Output.DICT)
			words = []
			for i in range(len(textline_tsv['level'])):
				if textline_tsv['conf'][i] != '-1':
					left = x_start + textline_tsv['left'][i]
					top = y_start + textline_tsv['top'][i]
					right = left + textline_tsv['width'][i]
					bottom = top + textline_tsv['height'][i]

					words.append({
							'word': textline_tsv['text'][i],
							'word_coord': [
											{'x': left, 'y': top}, 
											{'x': right, 'y': top}, 
											{'x': right, 'y': bottom}, 
											{'x': left, 'y': bottom}
										]
						})
			textline['words'] = words
		with open(json_path.replace('.json', '_ocr.json'), mode='w') as f:
			json.dump(obj, f)