### SIFTキーポイント 生成 サンプルコード

import cv2
from sift_matching_util import mask_keypoints, save_keypoints

print('しばらくお待ち下さい...')

img_query = cv2.imread('../query/query.png', cv2.IMREAD_GRAYSCALE)
keypts_query, desc_query = mask_keypoints(img_query, 10, (0.2, 1.0))
save_keypoints('../results/keypts_query.npz', keypts_query, desc_query)

img_sift = cv2.drawKeypoints(img_query, keypts_query, None, flags=4)
cv2.imwrite('../results/query_sift.png', img_sift)