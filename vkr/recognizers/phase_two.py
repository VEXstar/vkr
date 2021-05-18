import cv2
import numpy as np
from lungmask import mask
from vkr.loader.ct_loader import SIZE

model = mask.get_model('unet', 'LTRCLobes')


def get_lobes(sitk_obj, resize_fn):
    segmentation = mask.apply(sitk_obj, model)
    segmentation = list(segmentation)
    diff = 0
    classes = np.flip(np.delete(np.unique(segmentation), 0))
    start_lung = None
    end_lung = None
    for i in range(len(segmentation)):
        i = i - diff
        if len(np.unique(segmentation[i])) == 1:
            segmentation.pop(i)
            diff = diff + 1
            if start_lung is not None:
                end_lung = i + diff
        elif start_lung is None:
            start_lung = i + diff
    if end_lung is None:
        end_lung = len(segmentation) + diff
    f = []
    kernel = np.ones((2, 2), np.uint8)
    for i in range(len(segmentation)):
        resized = resize_fn(segmentation[i])
        cleared = np.zeros(shape=(SIZE, SIZE))
        for cl in classes:
            binar = 1 * (resized == cl)
            opening = cv2.morphologyEx(np.array(binar, dtype=np.uint8), cv2.MORPH_OPEN, kernel)
            cleared = cleared + cl * opening
        f.append(cleared)
    segmentation = np.array(f)
    return {'classes': classes, 'masks': segmentation, 'start': start_lung, 'end': end_lung}


def classes_dir_find(masks, classes):
    centre = masks.shape[2] / 2
    dir_class = {}
    range_lobes = {}
    for cl in classes:
        dir_class[cl] = {'l': 0, 'r': 0}
        range_lobes[cl] = []
    for i in range(len(masks)):
        for cl in classes:
            y, x = np.where(masks[i] == cl)
            if len(x) == 0:
                continue
            l_side = len(np.where(x < centre)[0])
            r_side = len(np.where(x > centre)[0])
            dir_class[cl]['l'] = dir_class[cl]['l'] + l_side
            dir_class[cl]['r'] = dir_class[cl]['r'] + r_side
            range_lobes[cl].append(i)
    for cl in range_lobes:
        lobe_r = range_lobes[cl]
        diff = 0
        for i in range(len(lobe_r) - 1):
            i = i - diff
            cur = lobe_r[i]
            next_lobe = lobe_r[i + 1]
            if abs(next_lobe - cur) <= 2:
                continue
            left = i
            right = len(lobe_r) - i
            if right > left:
                for f in range(i):
                    lobe_r.pop(f)
                diff = diff + left
            else:
                for f in range(left + 1, len(lobe_r)):
                    lobe_r.pop(f)
                diff = len(lobe_r) - left
    mins = {}
    maxs = {}
    for cl in range_lobes:
        maxs[cl] = max(range_lobes[cl])
        mins[cl] = min(range_lobes[cl])
    mins = dict(sorted(mins.items(), key=lambda item: item[1]))
    maxs = dict(sorted(maxs.items(), key=lambda item: item[1]))
    bazals_id = []
    for clmn in mins:
        step = int(len(classes) / 2)
        for clmx in maxs:
            if clmx == clmn:
                bazals_id.append(clmn)
                continue
            if step == 0:
                break
            step = step - 1
        if len(bazals_id) == 2:
            break
    response = {}
    for cl in classes:
        lr = dir_class[cl]
        if lr['l'] > lr['r']:
            response[cl] = 1
        elif lr['l'] < lr['r']:
            response[cl] = -1
        else:
            response[cl] = 0
    return {'bzl': bazals_id, 'dir': response, 'lobes_range': range_lobes}


def iou_for_lobes(last, cur):
    intersection = np.logical_and(last, cur)
    union = np.logical_or(last, cur)
    return np.sum(intersection) / np.sum(union)


def get_raw_info(lung_mask, seal_mask):
    lung_mask['masks'] = np.array(lung_mask['masks'])
    seal_mask = 1 * np.array(seal_mask)

    class_dir = classes_dir_find(lung_mask['masks'], lung_mask['classes'])

    lobe_count = {'lobe_conatins_seal': {}}
    for class_id in lung_mask['classes']:
        lobe_count[class_id] = 0
        lobe_count['lobe_conatins_seal'][class_id] = []
    lobe_count['total_inter'] = 0
    lobe_count['out_of_lung'] = 0
    lobe_count['out_of_lung_sclice'] = []
    lobe_count['in_lung_sclice'] = []
    lobe_count['diffusion'] = []
    lobe_count['warn_small_mask'] = True

    for i in range(len(seal_mask)):
        lobe_mask_ind = i - lung_mask['start']
        if (lobe_mask_ind < 0 or lobe_mask_ind >= len(lung_mask['masks'])) and len(np.unique(seal_mask[i])) > 1:
            lobe_count['out_of_lung'] = lobe_count['out_of_lung'] + 1
            lobe_count['out_of_lung_sclice'].append(i)
        elif not (lobe_mask_ind < 0 or lobe_mask_ind >= len(lung_mask['masks'])):
            lobe_count['in_lung_sclice'].append(i)
            lobes = lung_mask['masks'][lobe_mask_ind]
            seal = np.copy(seal_mask[i])
            if np.count_nonzero(seal) > 10:
                lobe_count['warn_small_mask'] = False
            mb_diff = []
            for cl in lung_mask['classes']:
                binar = lobes == cl
                inter = np.sum(np.logical_and(binar, seal)) / np.sum(binar)
                if inter > 0.01:
                    lobe_count[cl] = lobe_count[cl] + 1
                    lobe_count['total_inter'] = lobe_count['total_inter'] + 1
                    seal = seal - binar
                    mb_diff.append(cl)
                    lobe_count['lobe_conatins_seal'][cl].append(i)
            if len(mb_diff) > 1:
                lobe_count['diffusion'].append(mb_diff)
    lobe_count['out_of_lung'] = lobe_count['out_of_lung'] / len(seal_mask)
    lobe_count['attacted_precentage'] = np.count_nonzero(seal_mask) / np.count_nonzero(lung_mask['masks'])
    lobe_count['dir'] = class_dir
    return lobe_count


def interpretate_data(images, raw_data):
    formed_imgs = []
    report_text = []
    warns = []
    total_len = max(images.keys())
    norm_precent = round(raw_data['attacted_precentage'] * 100, 1)
    virus = 0
    steps = 0
    report_text.append("Повреждённый объем лёгких " + str(norm_precent)+"%.")
    if raw_data['out_of_lung'] > 0.1:
        out_p = round(raw_data['out_of_lung'] * 100, 1)
        warns.append("Системой были найдены уплотнения за предлеами лёгких,\n возможно положение уплотнений или "
                     "положение лёгких были определны неверно.\n "
                     " Процент от общего количества уплотннеий -  " + str(out_p) + "%.")
    bazal = raw_data['dir']['bzl']
    if raw_data[bazal[0]] > 0 or raw_data[bazal[1]] > 0:
        if raw_data[bazal[0]] > 0 and raw_data[bazal[1]] > 0:
            virus = virus + 1
            steps = steps + 1
            report_text.append("В базальной области лёгких обнаружены уплотнения с обеих сторон.")
        else:
            virus = virus + 0.6
            steps = steps + 1
            report_text.append("В базальной области лёгких обнаружены уплотнения с одной стороны.")
        prepare_obj = {'desc': "Базальная область", "slices": []}
        bzl_tmp = list(raw_data['lobe_conatins_seal'][bazal[0]])
        bzl_tmp.extend(raw_data['lobe_conatins_seal'][bazal[1]])
        bazal_seal = list(set(bzl_tmp))
        for sli in bazal_seal:
            prepare_obj['slices'].append({'numb': str(sli + 1) + '/' + str(total_len), 'img': images[sli]})
        formed_imgs.append(prepare_obj)
    left_sum = 0
    right_sum = 0
    total_lobes = len(raw_data['dir']['dir'])
    for side in raw_data['dir']['dir']:
        count_inter = raw_data[side]
        if raw_data['dir']['dir'][side] == -1:
            left_sum = left_sum + count_inter
        else:
            right_sum = right_sum + count_inter
        if count_inter > 0:
            total_lobes = total_lobes - 1
    smooth_diff = (min(left_sum, right_sum) + 1e-10) / (max(left_sum, right_sum) + 1e-10)  # переделать ?
    if smooth_diff > 0.1 and total_lobes == 0:
        virus = virus + 0.6
        steps = steps + 1
        report_text.append("Уплотнения расположены во всех отделах лёгих с обеих сторон.")
    elif smooth_diff > 0.1:
        virus = virus + 0.5
        steps = steps + 1
        report_text.append("Уплотнения локализуются с обеих сторон.")
    elif smooth_diff <= 0.1:
        virus = virus + 0.2
        steps = steps + 1
        report_text.append("Уплотнения приемущественно односторонние.")
    other_lobes = []
    for cli in raw_data['lobe_conatins_seal']:
        if cli in bazal:
            continue
        other_lobes.extend(raw_data['lobe_conatins_seal'][cli])
    other_lobes = list(set(other_lobes))
    prepare_obj = {'desc': "Сердняя и верхняя доли", "slices": []}
    for sli in other_lobes:
        prepare_obj['slices'].append({'numb': str(sli + 1) + '/' + str(total_len), 'img': images[sli]})
    formed_imgs.append(prepare_obj)
    prepare_obj = {'desc': "Срезы, которые не удалось отнести к отделам", "slices": []}
    for im_ind in images:
        in_b_l = im_ind in raw_data['lobe_conatins_seal'][bazal[0]]
        in_b_r = im_ind in raw_data['lobe_conatins_seal'][bazal[1]]
        if im_ind not in other_lobes and not in_b_l and not in_b_r:
            prepare_obj['slices'].append({'numb': str(im_ind + 1) + '/' + str(total_len), 'img': images[im_ind]})
    if len(prepare_obj['slices']) !=0:
        formed_imgs.append(prepare_obj)
    if raw_data['warn_small_mask'] or norm_precent < 2:
        warns.append("Размер уплотнений аномально малый, возможно найденные уплотнения таковыми не явлюятся.")

    if steps == 0:
        report_text.append("Вывод: уплотнения не были найдены или не удалось распознать положение лёгких.")
    else:
        virus = virus/steps
        bacterial = 1 - virus
        if bacterial > virus:
            report_text.append("Вывод: более вероятна бактериальная пневмония.")
        else:
            report_text.append("Вывод: более вероятна вирусная пневмония.")
        report_text.append("Процентное соотнощение принадлежности к классам:")
        report_text.append('бактериальная - ' + str(round(bacterial*100, 1)) + " вирусная - " + str(round(virus*100, 1)))
    if len(warns) > 0:
        report_text.append("ПРЕДУПРЕЖДЕНИЯ:")
        report_text.extend(warns)
    text = "\n".join(report_text)
    return text, formed_imgs


def merge_mask_and_ct(dataset, masks):
    images = {}
    real_index = 0
    for norm, real in dataset:
        for image in norm:
            seal_mask = 1 * masks[real_index]
            real_index = real_index + 1
            if len(np.unique(seal_mask)) == 1:
                continue
            img = image.numpy()
            seal_mask = np.squeeze(seal_mask, 0)
            img = np.rollaxis(img, 0, 3)
            seal_mask = cv2.normalize(seal_mask, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            gray_im = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            rgb_im = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2RGB)
            contours, hierarchy = cv2.findContours(seal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_im, contours, -1, (0, 255, 0), 1)
            images[real_index - 1] = rgb_im
    return images


def predict(data_loader, dataset, masks):
    images_raw = merge_mask_and_ct(data_loader, masks)
    lobes_mask = get_lobes(dataset.ct_obj, dataset.resize_image)
    raw_info = get_raw_info(lobes_mask, masks)
    print(raw_info)
    return interpretate_data(images_raw, raw_info)
