import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.feature import hog, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis, entropy

def segments_to_svm_input(segment_map, image, has_tooth_dict, cfg):
    inputs, labels = [], []
    
    for prop in regionprops(segment_map):
        # 0 to tło, pomijamy fizycznie nieistotne dane
        if prop.label == 0: 
            continue

        y1, x1, y2, x2 = prop.bbox
        # Wycinamy fragment z nałożoną maską segmentu (zero-out background)
        roi = image[y1:y2, x1:x2] * prop.image
        
        # Sprawdzanie minimalnego wymiaru z parametru cfg
        if roi.shape[0] < cfg['hog_min_dim'] or roi.shape[1] < cfg['hog_min_dim']: 
            continue
        
        # Konwersja do uint8 – krytyczna dla biblioteki HOG
        
        # Przeskalowanie do stałego rozmiaru wejściowego
        resized = cv2.resize(roi, cfg['hog_size'])
        
        # Właściwa ekstrakcja wektora HOG
        features = hog(
            resized, 
            orientations=cfg['hog_orientations'], 
            pixels_per_cell=cfg['hog_pixels_per_cell'], 
            cells_per_block=cfg['hog_cells_per_block'],
            block_norm=cfg.get('hog_block_norm', 'L2-Hys')
        )
        
        inputs.append(features)
        
        # Mapowanie etykiety ze słownika wygenerowanego wcześniej (truth matching)
        label_val = 1 if has_tooth_dict.get(prop.label, False) else 0
        labels.append(label_val)
        
    return np.array(inputs), np.array(labels)

def return_segments_with_teeth(image_object, segments):
    teeth_polys = get_teeth_positions(image_object)
    # Wyciągamy unikalne ID segmentów (np. 1, 2, 3... 32) pomijając TŁO (0)
    unique_ids = np.unique(segments)
    unique_ids = unique_ids[unique_ids != 0] 
    
    # Tworzymy słownik {id_segmentu: True/False}
    has_tooth_dict = {id_seg: False for id_seg in unique_ids}

    for seg_id in unique_ids:
        for poly in teeth_polys:
            # Szybka maska punktów zęba, które są wewnątrz obrazu
            p = poly[(poly[:, 0] >= 0) & (poly[:, 0] < segments.shape[1]) & 
                    (poly[:, 1] >= 0) & (poly[:, 1] < segments.shape[0])]
            
            # Jeśli WSZYSTKIE kropki metadanych danego zęba leżą w segmencie o danym ID
            if p.size > 0 and np.all(segments[p[:, 1].astype(int), p[:, 0].astype(int)] == seg_id):
                has_tooth_dict[seg_id] = True
                break
    return has_tooth_dict

def get_safe_stats(data_array):
    if data_array.size < 4 or np.std(data_array) < 1e-6:
        return 0.0, 0.0
    return float(skew(data_array)), float(kurtosis(data_array))

def get_teeth_positions(image_metadata_record: dict) -> list[np.ndarray]:
    teeth = []
    if 'objects' in image_metadata_record:
        for tooth_pos in image_metadata_record['objects']:
            teeth.append(np.array(tooth_pos['points']['exterior']))
    return teeth

def return_segments(preprocessed, cfg):
    # KSIZE=7 JEST KLUCZOWE
    x_deriv = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=cfg['sobel_ksize'])
    y_deriv = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=cfg['sobel_ksize'])

    y_range = np.arange(cfg['y_top_limit'], preprocessed.shape[0] - cfg['y_bot_limit'])
    y_center, img_w = y_range.mean(), preprocessed.shape[1]
    x_positions = np.arange(cfg['x_left_limit'], img_w - cfg['x_right_limit'], cfg['x_step'])
    angles = np.radians(np.linspace(-cfg['angle_range'], cfg['angle_range'], cfg['angle_steps']))

    node_costs = np.zeros((len(x_positions), len(angles)))
    for xi, x_pos in enumerate(x_positions):
        for ai, ang in enumerate(angles):
            line_x = np.clip(np.tan(ang)*(y_range - y_center) + x_pos, 0, img_w-1).astype(int)
            # PRZYWRÓCONE: Tylko abs(x_deriv) dla linii pionowych!
            grad_m = np.abs(x_deriv[y_range, line_x]).mean() 
            node_costs[xi, ai] = (cfg['alpha'] * preprocessed[y_range, line_x].mean() + 
                                cfg['beta'] * preprocessed[y_range, line_x].std() + 
                                cfg['gamma'] / (grad_m**2 + cfg['eps']) + 
                                cfg['delta'] * abs(ang))

    dp = np.full((cfg['num_planes'], len(x_positions), len(angles)), np.inf)
    backtrack = np.zeros_like(dp, dtype=int)
    dp[0, :len(x_positions)//4] = node_costs[:len(x_positions)//4]

    for p in range(1, cfg['num_planes']):
        for i in range(len(x_positions)):
            prev_idx = np.where(x_positions[i] - x_positions[:i] >= cfg['min_gap_pixels'])[0]
            if prev_idx.size == 0: 
                continue
            best_px = prev_idx[np.argmin(dp[p-1, prev_idx].min(axis=1))]
            dp[p, i, :] = dp[p-1, best_px].min() + node_costs[i, :]
            backtrack[p, i, :] = best_px

    curr_x = np.argmin(dp[-1].min(axis=1))
    lines = []
    for p in range(cfg['num_planes']-1, -1, -1):
        lines.append((np.tan(angles[0]) * (y_range - y_center) + x_positions[curr_x]).astype(int))
        curr_x = backtrack[p, curr_x, 0]

    # SEPARACJA POZIOMA (OCLLUSAL) - Tutaj magnitude jest ok
    mag = np.sqrt(x_deriv**2 + y_deriv**2)
    y_dn, y_up = int(preprocessed.shape[0] * cfg['y_occl_down']), int(preprocessed.shape[0] * cfg['y_occl_up'])
    roi_costs = cfg['alpha'] * preprocessed[y_dn:y_up, :] + cfg['gamma'] / (mag[y_dn:y_up, :]**2 + cfg['eps'])

    rows, cols = roi_costs.shape
    dp_h, bt_h = np.full((rows, cols), np.inf), np.zeros((rows, cols), dtype=int)
    dp_h[:, 0] = roi_costs[:, 0]
    
    for x in range(1, cols):
        L = dp_h[:, x-1]
        stack = np.stack([np.r_[L[1:], np.inf], L, np.r_[np.inf, L[:-1]]])
        best = np.argmin(stack, axis=0)
        dp_h[:, x] = roi_costs[:, x] + stack[best, np.arange(rows)]
        bt_h[:, x] = np.clip(np.arange(rows) + (best - 1), 0, rows-1)

    curve = np.zeros(cols, dtype=int)
    curve[-1] = np.argmin(dp_h[:, -1])
    for x in range(cols-1, 0, -1): # <--- TU BYŁ BŁĄD, TERAZ JEST COLS
        curve[x-1] = bt_h[curve[x], x]
    curve += y_dn

    # Generowanie finalnej mapy (używamy img_w z góry funkcji)
    label_map = np.zeros((len(y_range), img_w), dtype=np.uint8)
    for lx in lines: 
        label_map += (np.arange(img_w) > lx[:, None])
    
    y_coords, x_coords = np.indices((len(y_range), img_w))
    is_lower = (y_coords + cfg['y_top_limit']) > curve[x_coords]
    
    return label_map + (is_lower * 16)

def extract_dense_point_profile(points, roi_image, cfg):
    pts = np.float32(points).reshape(-1, 2)
    coords = pts.astype(int)
    h, w = roi_image.shape
    y_idx, x_idx = np.clip(coords[:, 1], 0, h-1), np.clip(coords[:, 0], 0, w-1)
    
    lx = cv2.Sobel(roi_image, cv2.CV_64F, 1, 0, ksize=cfg['sobel_ksize_local'])
    ly = cv2.Sobel(roi_image, cv2.CV_64F, 0, 1, ksize=cfg['sobel_ksize_local'])
    mag = np.sqrt(lx**2 + ly**2)
    
    intensities = roi_image[y_idx, x_idx].astype(float)
    gradients = mag[y_idx, x_idx]

    # Geometria
    centered = pts - pts.mean(axis=0)
    evals = np.sort(np.linalg.eigvals(np.cov(centered.T)))[::-1]
    dist_rad = np.linalg.norm(centered, axis=1)

    si, ki = get_safe_stats(intensities)
    sg, kg = get_safe_stats(gradients)
    sd, kd = get_safe_stats(dist_rad)

    glcm = graycomatrix(roi_image.astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
    hu = cv2.HuMoments(cv2.moments(pts.reshape(-1, 1, 2))).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + cfg['eps_hu'])

    return np.hstack([
        evals, evals[0]/(evals[1]+1e-6), np.mean(intensities), np.std(intensities), si, ki,
        np.percentile(intensities, [25, 50, 75]), np.mean(gradients), sg, kg,
        np.mean(dist_rad), np.std(dist_rad), sd, kd, hu_log,
        graycoprops(glcm, 'contrast')[0, 0], graycoprops(glcm, 'energy')[0, 0],
        entropy(np.histogram(intensities, 16, (0, 256))[0] + 1e-7)
    ]).astype(np.float32)