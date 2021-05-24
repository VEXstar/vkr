import cv2
import numpy as np
from torch.utils.data import DataLoader
from lungmask import mask
from vkr.loader.ct_loader import SIZE
import SimpleITK as sitk

model = mask.get_model('unet', 'LTRCLobes')

rgb_const = [(255, 0, 0), (0, 255, 0), (0, 217, 255), (255, 153, 0), (255, 153, 0), (183, 0, 255)]


class LungMaskInfo:
    def __init__(self, index, label, direction):
        self.index = index
        self.label = label
        self.dir = direction
        self.color = tuple(np.array(rgb_const[index]) / 255)

    def label_short_direction(self):
        if self.dir < 0:
            return 'l'
        return 'r'

    def get_fitz_color(self):
        return self.color

    def get_full_label(self):
        return self.label_direction() + " " + self.label

    def label_direction(self):
        if self.dir == 0:
            return ''
        if self.dir < 0:
            return 'Левая'
        return 'Правая'

    def get_label(self):
        return self.label


lung_mask_infos = {5: LungMaskInfo(5, "базальная", 1),
                   2: LungMaskInfo(2, "базальная", -1),
                   1: LungMaskInfo(1, "средняя и верхняя доли", -1),
                   3: LungMaskInfo(3, "средняя и верхняя доли", 1),
                   4: LungMaskInfo(4, 'средняя и верхняя доли 2', 1),
                   0: LungMaskInfo(0, 'Не удалось определить область', 0)
                   }


def get_lobes(dataset):
    not_hu = not dataset.is_hu()
    segmentation = []
    if not_hu:
        for (n, r) in DataLoader(dataset, batch_size=1):
            r_np = np.array(r)[0]
            image_rgb = sitk.GetImageFromArray(r_np)
            segmentation.append(mask.apply(image_rgb, model, batch_size=5, noHU=True)[0])
        segmentation = np.asarray(segmentation)
    else:
        segmentation = mask.apply(dataset.ct_obj, model, batch_size=5)
    classes = np.flip(np.delete(np.unique(segmentation), 0))
    f = []
    for i in range(len(segmentation)):
        f.append(dataset.resize_image(segmentation[i]))
    segmentation = f
    diff = 0
    start_lung = None
    end_lung = None
    for i in range(len(segmentation)):
        if len(np.unique(segmentation[i])) == 1:
            diff = diff + 1
            if start_lung is not None:
                end_lung = i
        elif start_lung is None:
            start_lung = i
        if start_lung is not None and end_lung is not None:
            break
    if end_lung is None:
        end_lung = len(segmentation) + diff
    return {'classes': classes,
            'masks': segmentation,
            'start': start_lung,
            'end': end_lung,
            }


def prepare_data(lung_mask, seal_mask):
    part_step = (lung_mask['end'] - lung_mask['start']) / 3
    lung_start = lung_mask['start']

    lung_mask = np.array(lung_mask['masks'])
    seal_mask = np.array(seal_mask)
    classes = np.unique(lung_mask)
    common_vol = np.zeros(6)
    common_lr_vol = np.zeros(2)
    result = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        'left_attacked': 0,
        'right_attacked': 0,
        'common_attacked': 0,
        'down_masked': np.zeros((len(seal_mask), 2, 2, SIZE, SIZE)),
        'centre_masked': np.zeros((len(seal_mask), 2, 2, SIZE, SIZE)),
        'up_masked': np.zeros((len(seal_mask), 2, 2, SIZE, SIZE)),
        'warn_small': True
    }

    for i in range(len(lung_mask)):
        mask = seal_mask[i]
        lung = lung_mask[i]
        for cl in classes:
            if cl == 0:
                continue

            bin_lobe = 1 * (np.array(lung) == cl)
            and_mask = 1 * np.logical_and(mask, bin_lobe)
            mask = mask - and_mask
            and_mask = np.squeeze(and_mask, 0)

            mask_vol = np.count_nonzero(and_mask)
            lobe_vol = np.count_nonzero(bin_lobe)
            if np.count_nonzero(mask_vol) == 0:
                continue
            if mask_vol > 32:
                result['warn_small'] = False

            result[cl] += mask_vol
            common_vol[cl] += lobe_vol

            lobe_info = lung_mask_infos[cl]

            if lobe_info.label_short_direction() == 'l':
                result['left_attacked'] += mask_vol
                common_lr_vol[0] += lobe_vol
            else:
                result['right_attacked'] += mask_vol
                common_lr_vol[1] += lobe_vol

            if i >= lung_start + part_step * 2:
                if lobe_info.label_short_direction() == 'l':
                    result['up_masked'][i][0][0] += and_mask
                    result['up_masked'][i][0][1] += bin_lobe
                else:
                    result['up_masked'][i][1][0] += and_mask
                    result['up_masked'][i][1][1] += bin_lobe
            elif i >= lung_start + part_step:
                if lobe_info.label_short_direction() == 'l':
                    result['centre_masked'][i][0][0] += and_mask
                    result['centre_masked'][i][0][1] += bin_lobe
                else:
                    result['centre_masked'][i][1][0] += and_mask
                    result['centre_masked'][i][1][1] += bin_lobe
            else:
                if lobe_info.label_short_direction() == 'l':
                    result['down_masked'][i][0][0] += and_mask
                    result['down_masked'][i][0][1] += bin_lobe
                else:
                    result['down_masked'][1][0] += and_mask
                    result['down_masked'][1][1] += bin_lobe
    losses_volume = result[0]
    common_volume = 1e-20
    for i in range(1, len(common_vol)):
        result[i] = (result[i] + 1e-20) / (common_vol[i] + 1e-20)
        common_volume += common_vol[i]
    temp_l = (result['left_attacked'] + 1e-20) / (common_lr_vol[0] + 1e-20)
    result['left_attacked'] = temp_l if temp_l != 1 else 0
    temp_r = (result['right_attacked'] + 1e-20) / (common_lr_vol[1] + 1e-20)
    result['right_attacked'] = temp_r if temp_r != 1 else 0
    result['common_attacked'] = (result['left_attacked'] + result['right_attacked'])/2 + losses_volume/common_volume
    return result


def normalize_pr(val):
    return round(val * 100, 2)


def iou_for_lobes(last, cur):
    intersection = np.logical_and(last, cur)
    union = np.logical_or(last, cur)
    return np.sum(intersection) / np.sum(union)


def defusion_find(masked):
    def_count = 0
    for i in range(1, len(masked)):
        left_count = np.count_nonzero(masked[i][0][0])
        right_count = np.count_nonzero(masked[i][1][0])
        common_count = left_count + right_count
        if common_count == 0:
            def_count = 0
            continue
        l_iou = iou_for_lobes(masked[i - 1][0][0], masked[i][0][0])
        r_iou = iou_for_lobes(masked[i - 1][1][0], masked[i][1][0])
        if l_iou > 0.01 or r_iou > 0.01:
            def_count += 1
        if def_count == 3:
            return True
    return False


def interpreter(pre_analyzed_data):
    report_text = []
    warns = []
    viral = 0
    bacterial = 0
    data = pre_analyzed_data
    step = 0

    report_text.append("Объем поражения левого лёгкого: " + str(normalize_pr(data['left_attacked'])) + '%.')
    report_text.append("Объем поражения правого лёгкого: " + str(normalize_pr(data['right_attacked'])) + '%.')
    report_text.append("Общий объем поражения (от объема лёгких): " + str(normalize_pr(data['common_attacked'])) + '%.')

    c_l = data['left_attacked']
    c_r = data['right_attacked']
    if c_l < 0.01 and c_r < 0.01:
        report_text.append("Процент поражения лёгких менее 1%, определить тип пневмонии невозможно.")
        if pre_analyzed_data[0] > 0.1:
            report_text.append("Найдено множество уплотнений за пределами лёгких, возможно,\n"
                               " лёгкие сегментированы неверно.")
        return "\n".join(report_text)

    down_def = defusion_find(data['down_masked'])
    centre_def = defusion_find(data['centre_masked'])
    up_def = defusion_find(data['up_masked'])
    count_def = np.count_nonzero(1 * np.array([down_def, centre_def, up_def]))

    if up_def:
        report_text.append("В нижних отделах наблюдаются диффузная локализация уплотнений.")
    if centre_def:
        report_text.append("В средних отделах наблюдаются диффузная локализация уплотнений.")
    if down_def:
        report_text.append("В верхних отделах наблюдаются диффузная локализация уплотнений.")

    c_p = min(c_l, c_r) / max(c_l, c_r)
    if data['common_attacked'] < 0.05:
        warns.append("Поражено менее 5% лёгких, возможно, выделенные уплотнения таковыми не являются.")
    if data['warn_small']:
        warns.append(
            "Размеры уплотнений относительно небольшие, возможно,\n выделенные уплотнения таковыми не являются.")

    if c_p > 0.2 and count_def >= 2:
        report_text.append("Поражения преимущественно двухсторонние, локализация диффузная.")
        viral += 0.9
        bacterial += 0.1
    elif c_p < 0.2 and count_def >= 2:
        report_text.append("Поражения преимущественно односторонние, локализация диффузная.")
        viral += 0.6
        bacterial += 0.4
    elif c_p > 0.2:
        report_text.append("Поражения преимущественно двухсторонние.")
        viral += 0.7
        bacterial += 0.3
    else:
        report_text.append("Поражения преимущественно односторонние.")
        viral += 0.2
        bacterial += 0.8
    step += 1

    down_l_attacked = (np.count_nonzero(data['down_masked'][0][0]) + 1e-20) / \
                      (np.count_nonzero(data['down_masked'][0][1]) + 1e-20)
    down_r_attacked = (np.count_nonzero(data['down_masked'][1][0]) + 1e-20) / \
                      (np.count_nonzero(data['down_masked'][1][1]) + 1e-20)
    down_l_attacked = down_l_attacked if down_l_attacked != 1 else 0
    down_r_attacked = down_r_attacked if down_r_attacked != 1 else 0
    down_prc = 0
    if max(down_l_attacked, down_r_attacked) != 0:
        down_prc = min(down_l_attacked, down_r_attacked) / max(down_l_attacked, down_r_attacked)

    if down_prc > 0.1:
        report_text.append("Поражения в нижних долях преимущественно двухсторонние.")
    else:
        report_text.append("Поражения в нижних долях преимущественно односторонние.")

    all_lobes_attacked = True
    for i in range(1, 6):
        if data[i] < 1e-20:
            all_lobes_attacked = False
            break
    down_count = np.count_nonzero(data['down_masked'][0][0]) + np.count_nonzero(data['down_masked'][1][0])
    centre_count = np.count_nonzero(data['centre_masked'][0][0]) + np.count_nonzero(data['centre_masked'][1][0])
    up_count = np.count_nonzero(data['up_masked'][0][0]) + np.count_nonzero(data['up_masked'][1][0])
    down_prc = (down_count + 1e-20) / (centre_count + up_count + 1e-20)
    down_prc = down_prc if down_prc != 1 else 0

    if all_lobes_attacked:
        report_text.append("Поражены все отделы лёгких.")
        viral += 0.75
        bacterial += 0.25
    elif not all_lobes_attacked and down_prc > 0.4:
        report_text.append("Поражены преимущественно нижние отделы.")
        viral += 0.75
        bacterial += 0.25
    else:
        viral += 0.2
        bacterial += 0.8
    step += 1

    is_zero = (down_count + up_count) == 0
    if not is_zero and centre_count / (down_count + up_count / 2) > 0.4 and count_def <= 1:
        report_text.append("Уплотнения локализуются преимущественно в средних отделах.")
        viral += 0.2
        bacterial += 0.8
        step += 1
    elif not is_zero and centre_count / (down_count + up_count / 2) > 0.4 and count_def > 1:
        report_text.append("Уплотнения локализуются преимущественно в средних отделах, локализация диффузная.")
        viral += 0.3
        bacterial += 0.7
        step += 1
    viral = viral / step
    bacterial = bacterial / step
    report_text.append("\n\nВероятностная принадлежность к типам пневмоний:")
    report_text.append("Вирусная - " + str(normalize_pr(viral)))
    report_text.append("Бактериальная - " + str(normalize_pr(bacterial)))
    if viral > bacterial:
        report_text.append("Более вероятна вирусная пневмония.")
    else:
        report_text.append("Более вероятна бактериальная пневмония.")
    if len(warns) > 0:
        report_text.append("ПРЕДУПРЕЖДЕНИЯ:")
        report_text.extend(warns)
    return "\n".join(report_text)


def mask_concatinator(lung_mask, seal_mask, real_images):
    g = []
    classes = np.unique(lung_mask)
    for i in range(len(real_images)):
        mask = seal_mask[i]
        lung = lung_mask[i]
        img = real_images[i]
        gray_im = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        rgb_im = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2RGB)
        for cl in classes:
            bin_lobe = 1 * (np.array(lung) == cl)
            and_mask = 1 * np.logical_and(mask, bin_lobe)
            mask = mask - and_mask
            and_mask = np.squeeze(and_mask, 0)
            temp_and = np.zeros((and_mask.shape[0], and_mask.shape[1], 3))
            for x in range(and_mask.shape[0]):
                for y in range(and_mask.shape[1]):
                    if and_mask[x][y] == 1:
                        temp_and[x][y] = list(rgb_const[cl])
            rgb_im = np.uint8(1 * rgb_im + 0.12 * temp_and)
        g.append(np.flip(rgb_im, 0))
    return g


def predict(dataset, masks):
    lobes_mask = get_lobes(dataset)
    images_raw = mask_concatinator(lobes_mask['masks'], masks, dataset.numpy())
    raw_info = prepare_data(lobes_mask, masks)
    text = interpreter(raw_info)
    print(raw_info)
    return {'text': text, 'images': images_raw, 'lobes': lung_mask_infos, 'start': lobes_mask['start'],
            'end': lobes_mask['end']}
