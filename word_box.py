import cv2
import json
import copy
import numpy as np

class WordBox(object):
	@staticmethod
	def _space_info(image, axis, cutoff=False):
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image_gray = cv2.normalize(image_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		image_gray = np.power(np.abs(image_gray), 2.5)
		image_gray = cv2.normalize(image_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

		image_inv = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)

		profile = image_inv.sum(axis=axis, dtype=int)
		if cutoff:
			threshold = profile.mean()
			profile[profile < threshold] = 0
		profile = np.append(profile, 1) # append an arbitrary non-zero number
		
		num_zero, start, stop = [], 0, 0
		for i in range(len(profile) - 1):
			if profile[i] != 0 and profile[i + 1] == 0:
				start = i + 1
			elif profile[i] == 0 and profile[i + 1] != 0:
				stop = i + 1
				num_zero.append([start, stop])
		return num_zero

	@staticmethod
	def _trunc(image, axis):
		space_info = WordBox._space_info(image, axis=axis)
		if len(space_info) == 0:
			left, right = 0, image.shape[1 - axis]
		else:
			left = space_info[0][1] if space_info[0][0] == 0 else 0
			right = space_info[-1][0] if space_info[-1][1] == image.shape[1 - axis] else image.shape[1 - axis]
		if left >= right:
			left, right = 0, image.shape[1 - axis]
		return left, right

	@staticmethod
	def _split_vertical(image, coord, threshold):
		start = min(coord, key=lambda c: [c['x'], c['y']])
		stop = max(coord, key=lambda c: [c['x'], c['y']])
		word_image = image[start['y']:stop['y'], start['x']:stop['x']]
		
		top_trunc, bottom_trunc = WordBox._trunc(word_image, axis=1)
		word_image_trunc = word_image[top_trunc:bottom_trunc]

		space_info = WordBox._space_info(word_image_trunc, axis=1)
		space_info.insert(0, [0, 0])
		space_info.append([word_image_trunc.shape[0], word_image_trunc.shape[0]])
		non_space = [space_info[i + 1][0] - space_info[i][1] for i in range(len(space_info) - 1)]

		top, bottom = 0, word_image_trunc.shape[0]
		spaces = [space for space in space_info if space[1] - space[0] > threshold]
		if len(spaces) == 1:
			if spaces[0][0] < bottom - spaces[0][1]:
				top = spaces[0][1]
			else:
				bottom = spaces[0][0]
		elif len(spaces) == 2:
			top, bottom = spaces[0][1], spaces[1][0]
		if start['y'] + top_trunc + top >= start['y'] + top_trunc + bottom:
			return start['y'], stop['y']
		return start['y'] + top_trunc + top, start['y'] + top_trunc + bottom
	
	@staticmethod
	def _split_horizontal(image, coord, threshold, mode):
		start = min(coord, key=lambda c: [c['x'], c['y']])
		stop = max(coord, key=lambda c: [c['x'], c['y']])
		word_image = image[start['y']:stop['y'], start['x']:stop['x']]
		left_trunc, right_trunc = WordBox._trunc(word_image, axis=0)
		word_image_trunc = word_image[:, left_trunc:right_trunc]

		space_info = WordBox._space_info(word_image_trunc, axis=0)
		spaces = [space for space in space_info if space[1] - space[0] > threshold]
		left, right = 0, word_image_trunc.shape[1]
		if len(spaces) == 1:
			if mode is None:
				if spaces[0][0] > 2 * (word_image_trunc.shape[1] - spaces[0][1]):
					left, right = 0, spaces[0][0]
				else:
					left, right = spaces[0][1], word_image_trunc.shape[1]
			if mode == 'r':
				left, right = spaces[0][1], word_image_trunc.shape[1]
			elif mode == 'l':
				left, right = 0, spaces[0][0]
		if start['x'] + left_trunc + left >= start['x'] + left_trunc + right:
			return start['x'], stop['x']
		return start['x'] + left_trunc + left, start['x'] + left_trunc + right

	@staticmethod
	def _threshold_v(image):
		space_info = WordBox._space_info(image, axis=1, cutoff=True)
		space_info.insert(0, [0, 0])
		space_info.append([image.shape[0], image.shape[0]])
		non_space = [space_info[i + 1][0] - space_info[i][1] for i in range(len(space_info) - 1)]
		threshold_v = max(non_space) * 0.28
		return threshold_v

	@staticmethod
	def _box(image, coord, threshold_h, mode):
		coord = copy.deepcopy(coord)
		start = min(coord, key=lambda c: [c['x'], c['y']])
		stop = max(coord, key=lambda c: [c['x'], c['y']])
		word_image = image[start['y']:stop['y'], start['x']:stop['x']]

		top, bottom = WordBox._split_vertical(image, coord, WordBox._threshold_v(word_image))
		coord[0]['y'] = top
		coord[1]['y'] = top
		coord[2]['y'] = bottom
		coord[3]['y'] = bottom
		left, right = WordBox._split_horizontal(image, coord, threshold=threshold_h, mode=mode)
		return left, top, right, bottom

	@staticmethod
	def box(image, coord, threshold_h, coord_after=None):
		left, top, right, bottom = WordBox._box(image, coord, threshold_h, mode=None)
		if coord_after is not None:
			after_l = WordBox._box(image, coord_after, threshold_h, mode='l')
			after_r = WordBox._box(image, coord_after, threshold_h, mode=None)
			if after_l != after_r:
				top = after_l[1] if after_l[1] < top else top
				bottom = after_l[3] if after_l[3] > bottom else bottom
				right = after_l[2]
		return left, top, right, bottom

	@staticmethod
	def word_box(json_path):
		with open(json_path, mode='r') as f:
			obj = json.load(f)

		image = cv2.imread(obj['image_dir'])

		for i, textline in enumerate(obj['text_lines']):
			start = min(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			stop = max(textline['coordinates'], key=lambda c: [c['x'], c['y']])
			textline_image = image[start['y']:stop['y'], start['x']:stop['x']]
			textline_coord = [{'x': start['x'], 'y': start['y']}, {'x': stop['x'], 'y': start['y']}, {'x': stop['x'], 'y': stop['y']}, {'x': start['x'], 'y': stop['y']}]
			
			top_trunc, bottom_trunc = WordBox._trunc(textline_image, axis=1)
			textline_image = textline_image[top_trunc: bottom_trunc]
			textline_top, textline_bottom = WordBox._split_vertical(image, textline_coord, WordBox._threshold_v(textline_image))
			space_info = WordBox._space_info(image[textline_top:textline_bottom, start['x']:stop['x']], axis=0)
			if len(space_info) > 1:
				space_info = np.asarray(space_info)
				space_info = space_info[:, 1] - space_info[:, 0]
				threshold_h = 2 * space_info.mean() * space_info.max() / (space_info.mean() + space_info.max())
			else:
				threshold_h = float('inf')
			words = textline['words']
			for i in range(len(words)):
				coord = words[i]['word_coord']
				coord_after = words[i + 1]['word_coord'] if i < len(words) - 1 else None
				
				coord_start = min(coord, key=lambda c: [c['x'], c['y']])
				coord_stop = max(coord, key=lambda c: [c['x'], c['y']])
				
				if coord_after:
					coord_after_start = min(coord_after, key=lambda c: [c['x'], c['y']])
					coord_after_stop = max(coord_after, key=lambda c: [c['x'], c['y']])
					if min(coord_after_stop['x'] - coord_after_start['x'], coord_after_stop['y'] - coord_after_start['y']) < 3:
						coord_after = None
				try:
					left, top, right, bottom = WordBox.box(image, coord, threshold_h, coord_after) if min(coord_stop['x'] - coord_start['x'], coord_stop['y'] - coord_start['y']) > 3 else (coord_start['x'], coord_start['y'], coord_stop['x'], coord_stop['y'])
				except:
					left, top, right, bottom = coord_start['x'], coord_start['y'], coord_stop['x'], coord_stop['y']
				words[i]['word_coord'] = [{'x': left, 'y': top}, {'x': right, 'y': top}, {'x': right, 'y': bottom}, {'x': left, 'y': bottom}]
		with open(json_path.replace('.json', '_word.json'), mode='w') as f:
			json.dump(obj, f)