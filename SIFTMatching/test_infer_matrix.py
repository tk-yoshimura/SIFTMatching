### SIFTキーポイント 透視変換行列推定 サンプルコード

import cv2
from sift_matching_util import load_keypoints, estimate_homography_from_keypoints

img_query = cv2.imread('../query/query.png', cv2.IMREAD_GRAYSCALE)
size = (img_query.shape[1], img_query.shape[0])
keypts_query, desc_query = load_keypoints('../results/keypts_query.npz')

sift, bf_matcher = cv2.SIFT_create(), cv2.BFMatcher()

for targetname in ['target_raw', 'target_noised', 'target_blocked', 'target_invalid', 'target_sobad', 'target_small']:
    print('Processing... %s' % targetname)

    img_target = cv2.imread('../target/%s.png' % targetname, cv2.IMREAD_GRAYSCALE)
    keypts_target, desc_target = sift.detectAndCompute(img_target, None)

    good, mat = estimate_homography_from_keypoints(keypts_query, desc_query, keypts_target, desc_target, bf_matcher)

    if mat is None:
        print('fail estimete transform')
        continue
    
    img_matches = cv2.drawMatchesKnn(img_query, keypts_query, img_target, keypts_target, good, None, flags=2)
    img_warp = cv2.warpPerspective(img_target, mat, size)

    cv2.imwrite('../results/%s_sift.png' % targetname, img_matches)
    cv2.imwrite('../results/%s_warp.png' % targetname, img_warp)