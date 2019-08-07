import cv2
import json
import editdistance
import numpy as np
import pytesseract
import xml.etree.ElementTree as ET

class Evaluation(object):
	@staticmethod
	def _textlines(bbox_path):
		'''
		Returns a list of textlines in bbox file specified by bbox_path. Each textline is a dict having x_start, y_start, x_stop, y_stop keys.
		Args:
			bbox_path (str): path to bbox file.
		'''
		def tag_type(tag):
			return tag.split('}')[-1]
		root = ET.parse(bbox_path).getroot()
		textlines = []
		for ele in root.iter():
			if tag_type(ele.tag) == 'TextLine':
				# Textline's text
				text = ele[1][0].text if ele[1][0].text is not None else ''
				# Textline's bounding box
				coords = ele[0].attrib['points']
				coords = coords.split()
				coords = np.asarray([list(map(lambda num : int(num), coord.split(','))) for coord in coords], dtype=int)
				x_start, y_start = coords.min(axis=0)
				x_stop, y_stop = coords.max(axis=0)
				# Append textline to the list
				textlines.append({'x_start': x_start, 'y_start': y_start, 'x_stop': x_stop, 'y_stop': y_stop, 'text': text})
		return textlines

	@staticmethod
	def _iou(boxA, boxB):
		'''
		Returns IoU between boxA and boxB. boxA and boxB are tuples having the form (left, top, right, bottom).
		'''
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
	 
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	 
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	 
		iou = interArea / float(boxAArea + boxBArea - interArea)
	 
		return iou

	@staticmethod
	def _calc_accuracy(ocr_list):
		'''
		Returns character and word accuracy calculated from pairs in ocr_list. Each pair is a dict having ground_truth and prediction keys.
		'''
		char_err, word_err, count = 0, 0, 0
		for pair in ocr_list:
			char_err += editdistance.eval(pair['ground_truth'], pair['prediction']) / len(pair['ground_truth'])
			word_err += editdistance.eval(pair['ground_truth'].split(), pair['prediction'].split()) / len(pair['ground_truth'].split())
			count += 1
		char_accuracy, word_accuracy = 1 - char_err / count, 1 - word_err / count
		return char_accuracy, word_accuracy
		
	@staticmethod
	def ground_truth(image_path, bbox_path, lang='vie'):
		'''
		Returns character and word accuracy calculated on the image specified by image_path and bbox_path.
		Args:
			image_path (str): path to image.
			bbox_path (str): path to bbox file.
			lang (str): Tesseract OCR language code.
		'''
		image = cv2.imread(image_path)
		ocr_list = []
		textlines = Evaluation._textlines(bbox_path)
		for textline in textlines:
			textline_image = image[textline['y_start']:textline['y_stop'], textline['x_start']:textline['x_stop']]
			prediction_text = pytesseract.image_to_string(textline_image, lang=lang, config='--psm 7 --oem 1')
			ocr_list.append({'ground_truth': textline['text'], 'prediction': prediction_text})

		return Evaluation._calc_accuracy(ocr_list)

	@staticmethod
	def prediction(json_path, bbox_path, lang='vie'):
		'''
		Returns character and word accuracy calculated from textline predictions in json file specified by json_path and bbox_path.
		Args:
			json_path (str): path to json file.
			bbox_path (str): path to bbox file.
			lang (str): Tesseract OCR language code.
		'''
		with open(json_path, mode='r') as f:
			obj = json.load(f)

		image = cv2.imread(obj['image_dir'])
		ocr_list = []
		textlines = Evaluation._textlines(bbox_path)
		for textline in obj['text_lines']:
			x, y, w, h = textline['x'], textline['y'], textline['w'], textline['h']
			x_start, y_start, x_stop, y_stop = x, y, x + w, y + h
			textline_image = image[y_start:y_stop, x_start:x_stop]

			prediction_text = pytesseract.image_to_string(textline_image, lang='vie', config='--psm 7 --oem 1')

			ground_truth_textline = max(textlines, key=lambda x: Evaluation._iou((x['x_start'], x['y_start'], x['x_stop'], x['y_stop']), (x_start, y_start, x_stop, y_stop)))
			ground_truth_text = ground_truth_textline['text']
			ocr_list.append({'ground_truth': ground_truth_text, 'prediction': prediction_text})

		return Evaluation._calc_accuracy(ocr_list)