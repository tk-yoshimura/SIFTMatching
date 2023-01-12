### アフィン変換テスト

import cv2
from sift_matching_util import affine_image

img = cv2.imread('../query/affine_test.png')

img_rot, _ = affine_image(img, scale=0.4, angle=30, borderValue=(128, 128, 64))
cv2.imwrite('../results/affine_test_1.png', img_rot)

img_rot, _ = affine_image(img, scale=1.6, angle=-40, borderValue=(128, 64, 128))
cv2.imwrite('../results/affine_test_2.png', img_rot)