### SIFTキーポイント 投影変換行列推定 サンプルコード

import cv2
from sift_matching_util import load_keypoints

img_query = cv2.imread('../query/query.png', cv2.IMREAD_GRAYSCALE)
keypts_query, desc_query = load_keypoints('../results/keypts_query.npz')

sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

for targetname in ['target_raw', 'target_noised', 'target_blocked']:
    print('Processing... %s' % targetname)

    img_target = cv2.imread('../target/%s.png' % targetname, cv2.IMREAD_GRAYSCALE)
    keypts_target, desc_target = sift.detectAndCompute(img_target, None)
    matches = bf.knnMatch(desc_query, desc_target, k=2)

    good = []
    for match, match_2nd in matches:
        if match.distance > 0.50 * match_2nd.distance:
            continue
                           
        good.append([match])

    img_matches = cv2.drawMatchesKnn(img_query, keypts_query, img_target, keypts_target, good, None, flags=2)

    cv2.imwrite('../results/%s_sift.png' % targetname, img_matches)