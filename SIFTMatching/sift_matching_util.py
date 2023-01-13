### SIFTキーポイントユーティリティ

import warnings, itertools
import cv2
import numpy as np

"""
    拡大縮小/回転行列と変形後のサイズを求める
    Arguments:
        size((int, int)): 画像サイズ
        scale(float): 拡大率
        angle(float): 回転角[degree]
    Returns:
        mat: 変形行列
        size: 変形後サイズ
"""
def affine_matrix(size: tuple, scale: float, angle: float):
    w, h = size
    rad = angle * np.pi / 180

    w_rot = int(np.round(scale * (w * np.abs(np.cos(rad)) + h * np.abs(np.sin(rad)))))
    h_rot = int(np.round(scale * (w * np.abs(np.sin(rad)) + h * np.abs(np.cos(rad)))))
        
    size_rot = (w_rot, h_rot)
        
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale).copy()
    mat[:, 2] += (size_rot - np.array([w, h])) / 2

    return mat, size_rot

"""
    画像をはみ出ないよう拡大縮小/回転します
    Arguments:
        img(numpy.ndarray): 画像
        scale(float): 拡大率
        angle(float): 回転角[degree]
        borderValue((int, int, int)): 範囲外補間値
    Returns:
        img_rot: 変形画像
"""
def affine_image(img: np.ndarray, scale: float, angle: float, borderValue: tuple=(0, 0, 0)) -> np.ndarray:
    assert scale > 0, 'invalid scale.'
    
    while scale < 0.5:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//2, h//2), cv2.INTER_AREA)
        scale *= 2
            
    h, w = img.shape[:2]
    
    mat, size = affine_matrix((w, h), scale, angle)

    if scale < 1:
        img_rot = cv2.warpAffine(img, mat, size, flags=cv2.INTER_AREA, borderValue=borderValue)
    else:
        img_rot = cv2.warpAffine(img, mat, size, flags=cv2.INTER_LINEAR, borderValue=borderValue)

    return img_rot

"""
    拡大縮小/回転に対しロバストなキーポイントを抽出します
    Arguments:
        img(numpy.ndarray): 画像
        scale_range((float, float)): テンプレート拡大率範囲(<1)
        angle_range(int): テンプレート傾き範囲
        adopt_threshold(float): 座標ズレしきい値
        max_points(int): 最大キーポイント数
    Returns:
        keypts: キーポイント
        desc: キーポイント特徴量
"""
def mask_keypoints(img: np.ndarray, angle_range: int, scale_range: tuple, adopt_threshold:float = 4.0, max_points:int = 512):
    assert not img is None and img.ndim == 2, 'invalid img.'
    assert type(angle_range) is int and angle_range >= 0, 'angle_range'
    assert len(scale_range) == 2 and scale_range[0] <= scale_range[1], 'invalid scale_range.'

    if scale_range[1] < 1:
        warnings.warn('Deprecation: max_scale < 1')

    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

    keypts, desc = sift.detectAndCompute(img, None)
    coord = cv2.KeyPoint_convert(keypts)
    scores = np.zeros(len(coord), dtype=int)

    h, w = img.shape[:2]

    for angle, scale in itertools.product(range(-angle_range, angle_range + 1, 1), np.arange(scale_range[0], scale_range[1] + 1e-8, 0.05)):

        img_train = affine_image(img, scale, angle, borderValue=(128, 128, 128))
        mat, _ = affine_matrix((w, h), scale, angle)

        keypts_train, desc_train = sift.detectAndCompute(img_train, None)
            
        matches = bf.knnMatch(desc, desc_train, k=2)
        indexes_query = np.full(len(coord), -1, dtype=int)
            
        for match, match_2nd in matches:
            if match.distance > 0.75 * match_2nd.distance:
                continue
                
            indexes_query[match.queryIdx] = match.trainIdx      
                
        indexes_train = np.where(indexes_query >= 0)[0]
        coord_query = coord[indexes_query >= 0]
        coord_train = cv2.KeyPoint_convert(keypts_train, indexes_query[indexes_query >= 0])
        
        coord_affine = np.matmul(coord_query, mat[:, :2].T) + mat[:, 2]

        error = np.sum(np.square(coord_train - coord_affine), axis=1)

        indexes_adopt = indexes_train[error < (adopt_threshold * adopt_threshold)]
        scores[indexes_adopt] += 1

    scores_adopt = np.argsort(scores)[::-1]
    if len(scores_adopt) > max_points:
        scores_adopt = scores_adopt[:max_points]

    scores_adopt = scores_adopt[scores[scores_adopt] > 0]

    keypts_adopt = [keypts[index] for index in scores_adopt]
    keypts_adopt = tuple(keypts_adopt) if len(keypts_adopt) > 0 else ()

    desc_adopt = [desc[index] for index in scores_adopt]
    desc_adopt = np.stack(desc_adopt, axis=0) if len(desc_adopt) > 0 else np.zeros((0, desc.shape[1]), desc.dtype) 

    return keypts_adopt, desc_adopt

"""
    キーポイントをファイルに保存します
    Arguments:
        filepath(str): ファイルパス
        keypts: キーポイント
        desc: キーポイント特徴量
        compressed(bool): バイナリを圧縮するか
        precision(str): 浮動小数点精度 ('float', 'double')
"""
def save_keypoints(filepath: str, keypts, desc, compressed: bool= True, precision: str='double'):
    assert len(keypts) > 0, 'invalid counts'
    assert len(keypts) == len(desc), 'mismatch counts'
    assert precision in ['float', 'double'], 'invalid precision'

    pts, angles, responses, sizes, class_ids, octaves = [], [], [], [], [], []

    for keypt in keypts:
        pts.append(list(keypt.pt))
        angles.append(keypt.angle)
        responses.append(keypt.response)
        sizes.append(keypt.size)
        class_ids.append(keypt.class_id)
        octaves.append(keypt.octave)

    pts = np.array(pts)
    sizes = np.stack(sizes)
    angles = np.stack(angles)
    responses = np.stack(responses)
    octaves = np.stack(octaves)
    class_ids = np.stack(class_ids)

    if precision == 'float':
        pts = pts.astype(np.float32)
        sizes = sizes.astype(np.float32)
        angles = angles.astype(np.float32)
        responses = responses.astype(np.float32)

    if compressed:
        np.savez_compressed(filepath, pt=pts, size=sizes, angle=angles, response=responses, octave=octaves, class_id=class_ids, desc=desc)
    else:
        np.savez(filepath, pt=pts, size=sizes, angle=angles, response=responses, octave=octaves, class_id=class_ids, desc=desc)

"""
    キーポイントをファイルから読み込みます
    Arguments:
        filepath(str): ファイルパス
    Returns:
        keypts: キーポイント
        desc: キーポイント特徴量
"""
def load_keypoints(filepath: str):
    data = np.load(filepath)

    desc = data['desc']

    keypts = []
    for pt, size, angle, response, octave, class_id in zip(data['pt'], data['size'], data['angle'], data['response'], data['octave'], data['class_id']):
        keypt = cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle, response=response, octave=octave, class_id=class_id)

        keypts.append(keypt)

    assert len(keypts) > 0, 'invalid counts'
    assert len(keypts) == len(desc), 'mismatch counts'

    keypts = tuple(keypts)
    
    return keypts, desc

"""
    ターゲットからクエリへ透視変換する行列をキーポイント対応から推定する
    Arguments:
        keypts_query: キーポイント(クエリ)
        desc_query: キーポイント特徴量(クエリ)
        keypts_target: キーポイント(ターゲット)
        desc_target: キーポイント特徴量(ターゲット)
        matcher: キーポイント対応付けマッチング手法
        distance_thr(float): 特徴量スコア 第2候補/第1候補 比 ([0, 1])
                             小さいほど判定が厳しい
        min_points(int): 最小対応点数
    Returns:
        good: キーポイント対応
        mat: 変換行列
             候補が見つからなければNoneを返す
"""
def estimate_homography_from_keypoints(keypts_query, desc_query, keypts_target, desc_target, matcher, distance_thr:float=0.5, min_points:int=8):
    assert distance_thr >= 0 and distance_thr <= 1, 'invalid distance_thr'
    assert min_points >= 4, 'invalid min_points'

    matches = matcher.knnMatch(desc_query, desc_target, k=2)

    good, query_pts, target_pts = [], [], []
    for match, match_2nd in matches:
        if match.distance > distance_thr * match_2nd.distance:
            continue
        
        good.append([match])
        query_pts.append(list(keypts_query[match.queryIdx].pt))
        target_pts.append(list(keypts_target[match.trainIdx].pt))

    if len(good) < min_points:
        return good, None
        
    query_pts, target_pts = np.array(query_pts), np.array(target_pts)

    try:
        mat, mask = cv2.findHomography(target_pts, query_pts, cv2.RANSAC, 5)
    except:
        return good, None

    indexes_adopts = np.where(mask[:, 0] > 0)[0]
    
    adopts = [good[index] for index in indexes_adopts]

    if len(adopts) < min_points:
        return adopts, None

    return adopts, mat