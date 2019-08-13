import cv2
import json
import numpy as np

def _space_info(image, axis, cutoff=False):
    image_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # neu la chu den nen trang
    image_binary_inv = cv2.adaptiveThreshold(image_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    # ne la chu trang nen den
    # khong can inv TODO
    profile = image_binary_inv.sum(axis=axis, dtype=int)
    if cutoff:
        profile[profile < profile.mean()] = 0
    profile = np.append(profile, 1) # append an arbitrary non-zero number
    num_zero, start, stop = [], 0, 0
    for i in range(len(profile) - 1):
        if profile[i] != 0 and profile[i + 1] == 0:
            start = i + 1
        elif profile[i] == 0 and profile[i + 1] != 0:
            stop = i + 1
            num_zero.append([start, stop])
    return num_zero

def _trunc(image, axis):
    space_info = _space_info(image, axis=axis)
    if len(space_info) == 0:
        left, right = 0, image.shape[1 - axis]
    else:
        left = space_info[0][1] if space_info[0][0] == 0 else 0
        right = space_info[-1][0] if space_info[-1][1] == image.shape[1 - axis] else image.shape[1 - axis]
    return left, right

def _split_horizontal(image, coord, threshold, mode=None):
    start = min(coord, key=lambda c: [c['x'], c['y']])
    stop = max(coord, key=lambda c: [c['x'], c['y']])
    word_image = image[start['y']:stop['y'], start['x']:stop['x']]
    
    left_trunc, right_trunc = _trunc(word_image, axis=0)
    word_image_trunc = word_image[:, left_trunc:right_trunc]
    
    space_info = _space_info(word_image_trunc, axis=0)
    spaces = [space for space in space_info if space[1] - space[0] > threshold]
    left, right = 0, word_image_trunc.shape[1]
    # if len(spaces) == 0:
    #     if mode == 'l':
    #         left, right = 0, 0
    if len(spaces) == 1:
    	# if mode is None:
    	# 	if spaces[0][0] < word_image_trunc.shape[1] - spaces[0][1]:
    	# 		left, right = spaces[0][1], word_image_trunc.shape[1]
    	# 	else:
    	# 		left, right = 0, spaces[0][0]
        if mode == 'r':
            left, right = spaces[0][1], word_image_trunc.shape[1]
        elif mode == 'l':
            left, right = 0, spaces[0][0]

    return start['x'] + left_trunc + left, start['x'] + left_trunc + right

def _split_vertical(image, coord, threshold):
    start = min(coord, key=lambda c: [c['x'], c['y']])
    stop = max(coord, key=lambda c: [c['x'], c['y']])
    word_image = image[start['y']:stop['y'], start['x']:stop['x']]
    
    top_trunc, bottom_trunc = _trunc(word_image, axis=1)
    word_image_trunc = word_image[top_trunc:bottom_trunc]
    
    space_info = _space_info(word_image_trunc, axis=1)
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
    
    return start['y'] + top_trunc + top, start['y'] + top_trunc + bottom

def _box(image, coord, threshold_h, mode):
    start = min(coord, key=lambda c: [c['x'], c['y']])
    stop = max(coord, key=lambda c: [c['x'], c['y']])
    word_image = image[start['y']:stop['y'], start['x']:stop['x']]
    
    space_info = _space_info(word_image, axis=1, cutoff=True)
    space_info.insert(0, [0, 0])
    space_info.append([word_image.shape[0], word_image.shape[0]])
    non_space = [space_info[i + 1][0] - space_info[i][1] for i in range(len(space_info) - 1)]
    threshold_v = max(non_space) * 0.3

    top, bottom = _split_vertical(image, coord, threshold_v)
    coord[0]['y'] = top
    coord[1]['y'] = top
    coord[2]['y'] = bottom
    coord[3]['y'] = bottom
    left, right = _split_horizontal(image, coord, threshold=threshold_h, mode=mode)
    return left, top, right, bottom

def box(image, coord, threshold_h, coord_after=None):
    left, top, right, bottom = _box(image, coord, threshold_h, mode='r')
    if coord_after is not None:
        after_l = _box(image, coord_after, threshold_h, mode='l')
        after_r = _box(image, coord_after, threshold_h, mode='r')
        if after_l != after_r:
            right = after_l[2]
    return left, top, right, bottom