import os

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

from .text_render import get_char_glyph, put_char_horizontal, add_color
from .ballon_extractor import extract_ballon_region
from ..utils import TextBlock, rect_distance

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', "\""}


class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def render_lines(
    textlines: List[Textline],
    canvas_h: int,
    canvas_w: int,
    font_size: int,
    stroke_width: int,
    line_spacing: int = 0.01,
    fg: Tuple[int] = (0, 0, 0),
    bg: Tuple[int] = (255, 255, 255)) -> Image.Image:

    # bg_size = int(max(font_size * 0.1, 1)) if bg is not None else 0
    bg_size = stroke_width
    spacing_y = int(font_size * (line_spacing or 0.01))

    # make large canvas
    canvas_w = max([l.length for l in textlines]) + (font_size + bg_size) * 2
    canvas_h = font_size * len(textlines) + spacing_y * (len(textlines) - 1)  + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [font_size + bg_size, font_size + bg_size]

    # write stuff
    for line in textlines:
        pen_line = pen_orig.copy()
        pen_line[0] += line.pos_x # center
        for c in line.text:
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, width, height = cv2.boundingRect(canvas_border)
    return Image.fromarray(line_box[y:y+height, x:x+width])

    # c = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
    # d = ImageDraw.Draw(c)
    # d.fontmode = 'L'
    # for line in lines:
    #     d.text((line.pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=font_size, stroke_fill=stroke_color)
    # return c

def seg_eng(text: str) -> List[str]:
    """
    Extracts every word from text parameter
    """
    # TODO: replace with regexes

    text = text.strip().upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to ensure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
        if c in PUNSET_RIGHT_ENG and ii < text_len - 1:
            next_c = text[ii + 1]
            if next_c.isalpha() or next_c.isnumeric():
                processed_text += c + ' '
            else:
                processed_text += c
        else:
            processed_text += c

    word_list = processed_text.split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if len(word) < 3:
            append_left, append_right = False, False
            len_word, len_next, len_prev = len(word), -1, -1
            if ii < word_num - 1:
                len_next = len(word_list[ii + 1])
            if ii > 0:
                len_prev = len(words[-1])
            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1
            if len_next > 0 and len_prev > 0:
                if len_next < len_prev:
                    append_right = cond_next
                else:
                    append_left = cond_prev
            elif len_next > 0:
                append_right = cond_next
            elif len_prev:
                append_left = cond_prev

            if append_left:
                words[-1] = words[-1] + ' ' + word
            elif append_right:
                words.append(word + ' ' + word_list[ii + 1])
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)
    return words

def layout_lines_aligncenter(
    mask: np.ndarray, 
    words: List[str], 
    word_lengths: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False)->List[Textline]:
    """在气泡掩码内以质心为中心，将单词居中折行排版。"""

    # 计算掩码质心作为排版锚点；反转掩码使非零区域表示不可排版区域（边框/外部）
    m = cv2.moments(mask)
    mask = 255 - mask
    centroid_y = int(m['m01'] / m['m00'])
    centroid_x = int(m['m10'] / m['m00'])

    # 选取几何中心词：累加宽度最接近整句中点的单词，其余拆成左右两组
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    if num_words > 1:
        wl_array = np.array(word_lengths, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = np.argmin(np.abs(wl_cumsums))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = word_lengths[:central_index]
            sum_left = np.sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = word_lengths[central_index + 1:]
            sum_right = np.sum(len_right)
    else:
        central_index = 0

    # 将中心词放在质心位置，作为中心行的起点
    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - word_lengths[central_index] // 2

    bh, bw = mask.shape[:2]
    central_line = Textline(words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
    line_bottom = pos_y + line_height

    # 从中心行向左右两侧扩词：检测行边界是否越界或碰到障碍物
    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False

        # 尝试在左侧追加一个词
        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                if mask[pos_y: line_bottom, new_x_l].sum()==0 and mask[pos_y: line_bottom, new_r_l].sum() == 0:
                    left_valid = True
        # 尝试在右侧追加一个词
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if (new_x_r > 0 and new_r_r < bw):
                if mask[pos_y: line_bottom, new_x_r].sum()==0 and mask[pos_y: line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        # 两侧都能加时优先扩剩余总宽度更大的一侧；都加不了则停止
        insert_left = False
        if left_valid and right_valid:
            if sum_left > sum_right:
                insert_left = True
        elif left_valid:
            insert_left = True
        elif not right_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r
        if central_line.length > max_central_width:
            break

    # 得到中心行
    central_line.strip_spacing()
    lines = [central_line]

    # 中心行右侧剩余单词：从质心下方开始向下折行，放不下就往下换行
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_right > 0:
                        w, wl = wlst_right.pop(0), len_right.pop(0)
                        sum_right -= wl
                    else:
                        line.strip_spacing()
                        break

            # 当前行放不下则换到下一行，新行仍水平居中
            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # 中心行左侧剩余单词：从质心上方开始向上折行，放不下就往上换行
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_left > 0:
                        w, wl = wlst_left.pop(-1), len_left.pop(-1)
                        sum_left -= wl
                    else:
                        line.strip_spacing()
                        break

            # 当前行放不下则换到上一行，新行仍水平居中
            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
    # for line in lines:
    #     cv2.rectangle(rbgmsk, (line.pos_x, line.pos_y), (line.pos_x + line.length, line.pos_y + line_height), (0, 255, 0))
    # cv2.imshow('mask', rbgmsk)
    # cv2.waitKey(0)

    return lines

def render_textblock_list_eng(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_color = (0, 0, 0),
    stroke_color = (255, 255, 255),
    delimiter: str = ' ',
    line_spacing: int = 0.01,
    stroke_width: float = 0.1,
    size_tol: float = 1.0,
    ballonarea_thresh: float = 2,
    downscale_constraint: float = 0.7,
    original_img: np.ndarray = None,
    disable_font_border: bool = False,
    verbose: bool = False,
) -> np.ndarray:

    r"""
    将英文译文按气泡形状排版，并绘制到图像上。

    Args:
        stroke_width: 描边占字号的比例
        downscale_constraint (float, optional): 字号缩小下限，防止渲染文字过小
        ref_textballon (bool, optional): 是否以气泡轮廓作为排版参考
        original_img (np.ndarray, optional): 用于提取气泡区域的原图
    """

    # 按给定字号计算描边宽度、行高、分隔符宽度，最宽单词宽度，每个单词的像素宽度
    def calculate_font_values(font_size: int, words: List[str]):
        font_size = int(font_size)
        # 这里 font_size 单位像素，计算描边占用的像素
        sw = int(font_size * stroke_width)
        line_height = int(font_size * 0.8)
        # 分隔符宽度
        delimiter_glyph = get_char_glyph(delimiter, font_size, 0)
        delimiter_len = delimiter_glyph.advance.x >> 6
        base_length = -1  # 最宽单词的宽度，判断字号要不要缩小
        word_lengths = []
        for word in words:
            word_length = 0
            for cdpt in word:
                glyph = get_char_glyph(cdpt, font_size, 0)
                # 水平前进量：画完这个字符后，光标在 x 方向应前进多少。真实值 = 整数 / 64
                char_offset_x = glyph.metrics.horiAdvance >> 6
                word_length += char_offset_x
            word_lengths.append(word_length)
            if word_length > base_length:
                base_length = word_length
        return font_size, sw, line_height, delimiter_len, base_length, word_lengths

    # 转为 PIL 图像，后续用 paste 叠加文字图层
    img_pil = Image.fromarray(img)

    # 初始化每个文本块的气泡放大比例，以及放大后的包围盒
    for region in text_regions:
        region.enlarge_ratio = 1
        region.enlarged_xyxy = region.xyxy.copy()

    # 按 enlarge_ratio 以中心向外扩展文本块包围盒
    def update_enlarged_xyxy(region):
        region.enlarged_xyxy = region.xyxy.copy()
        # 计算放大后宽高相对原宽高的diff，将原box从中心向四周放大
        w_diff, h_diff = ((region.xywh[2:] * region.enlarge_ratio) - region.xywh[2:].astype(np.float64)) // 2
        region.enlarged_xyxy[0] -= w_diff
        region.enlarged_xyxy[2] += w_diff
        region.enlarged_xyxy[1] -= h_diff
        region.enlarged_xyxy[3] += h_diff

    # 按长宽比估算放大比例；若放大后与相邻块相交，则按原始间距回缩双方比例，避免重叠
    for region in text_regions:
        # 尚未被相交调整过时，长宽比越大，越倾向于放大气泡
        if region.enlarge_ratio == 1:
            # 放大比例，min(1.5*宽高比，3)
            region.enlarge_ratio = min(max(region.xywh[2] / region.xywh[3], region.xywh[3] / region.xywh[2]) * 1.5, 3)
            update_enlarged_xyxy(region)

        for region2 in text_regions:
            if region is region2:
                continue

            # 放大后的包围盒相交时，按原始间距重新分配双方放大比例
            if rect_distance(*region.enlarged_xyxy, *region2.enlarged_xyxy) == 0:
                d = rect_distance(*region.xyxy, *region2.xyxy)
                l1 = (region.xywh[2] + region.xywh[3]) / 2
                l2 = (region2.xywh[2] + region2.xywh[3]) / 2
                region.enlarge_ratio = d / (2 * l1) + 1
                region2.enlarge_ratio = d / (2 * l2) + 1
                update_enlarged_xyxy(region)
                update_enlarged_xyxy(region2)
                # print('Reducing enlarge ratio to prevent intersection')
                # print(region.translation, region.enlarged_xyxy, region.enlarge_ratio)
                # print('>->', region2.translation, region2.enlarged_xyxy, region2.enlarge_ratio)

    # 逐个文本块：分词、提取气泡、排版并绘制
    for region in text_regions:
        # 将译文切成英文单词列表；空译文直接跳过
        words = seg_eng(region.translation)
        if not words:
            continue

        # 按当前字号预计算各单词宽度等排版参数
        font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(region.font_size, words)

        # 从原图提取对应气泡掩码及其外接框（非深度学习分割）
        ballon_mask, xyxy = extract_ballon_region(original_img, region.xywh, enlarge_ratio=region.enlarge_ratio, verbose=verbose)
        # mask中可用区域（气泡内部）的像素
        ballon_area = (ballon_mask > 0).sum()
        # rx、ry 是旋正气泡掩码后，坐标原点相对原裁剪区域的 x/y 偏移量
        rotated, rx, ry = False, 0, 0

        # 倾斜超过 3 度时，把气泡掩码旋正以便水平排版，并记录旋转带来的原点偏移 (rx, ry)
        if abs(region.angle) > 3:
            rotated = True
            region_angle_rad = np.deg2rad(region.angle)
            region_angle_sin = np.sin(region_angle_rad)
            region_angle_cos = np.cos(region_angle_rad)
            rotated_ballon_mask = Image.fromarray(ballon_mask).rotate(region.angle, expand=True)
            rotated_ballon_mask = np.array(rotated_ballon_mask)

            region.angle %= 360
            if region.angle > 0 and region.angle <= 90:
                ry = abs(ballon_mask.shape[1] * region_angle_sin)
            elif region.angle > 90 and region.angle <= 180:
                rx = abs(ballon_mask.shape[1] * region_angle_cos)
                ry = rotated_ballon_mask.shape[0]
            elif region.angle > 180 and region.angle <= 270:
                ry = abs(ballon_mask.shape[0] * region_angle_cos)
                rx = rotated_ballon_mask.shape[1]
            else:
                rx = abs(ballon_mask.shape[0] * region_angle_sin)
            ballon_mask = rotated_ballon_mask

        # 估算「单词全部铺成一行」所需面积，并与气泡面积比较，得到缩放比（当前按面积缩放的逻辑已注释）
        line_width = sum(word_lengths) + delimiter_len * (len(word_lengths) - 1)
        # line_width * line_height 已经算出一行单词面积，后面delimiter_len算的是附加面积，用于留白余量
        region_area = line_width * line_height + delimiter_len * (len(words) - 1) * line_height
        area_ratio = ballon_area / region_area
        resize_ratio = 1

        # 气泡面积过小时放大掩码；实际使用中常把字号缩得过小，故已禁用
        # # if ballon_area is smaller than 2*region_area
        # if area_ratio < ballonarea_thresh:
        #     # resize so that it is 2*region_area
        #     resize_ratio = ballonarea_thresh / area_ratio
        #     ballon_area = int(resize_ratio * ballon_area) # = ballonarea_thresh * line_area
        #     resize_ratio = min(np.sqrt(resize_ratio), (1/downscale_constraint)**2)
        #     rx *= resize_ratio
        #     ry *= resize_ratio
        #     ballon_mask = cv2.resize(ballon_mask, (int(resize_ratio * ballon_mask.shape[1]), int(resize_ratio * ballon_mask.shape[0])))

        # 求气泡掩码的外接矩形，作为可用排版区域
        region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(ballon_mask))

        # 按气泡宽度和可容纳行数估算字号缩放系数；放不下则缩小字号并重算宽度，但不低于 downscale_constraint
        # base_length_word 长度最长的单词
        base_length_word = words[max(enumerate(word_lengths), key = lambda x: x[1])[0]]
        if len(base_length_word) == 0 :
            continue
        # 估计需要几行：译文字符数 / 最宽单词字符数
        lines_needed = len(region.translation) / len(base_length_word)
        # 估计能放几行：裁剪窗口高度 / 行高
        lines_available = abs(xyxy[3] - xyxy[1]) // line_height + 1
        # 宽度约束：气泡掩码外接矩形宽 / (最宽大词宽度 + 左右描边)
        # 高度约束：能放的行数 / 需要的行数
        font_size_multiplier = max(min(region_w / (base_length + 2*sw), lines_available / lines_needed), downscale_constraint)
        # 如果需要的宽度或高度超出了能提供的，则缩小字号，并重新计算排版参数
        if font_size_multiplier < 1:
            font_size = int(font_size * font_size_multiplier)
            font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(font_size, words)

        # 在气泡掩码内按中心对齐把单词折成多行
        textlines = layout_lines_aligncenter(ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter)

        # 将整段文字在竖直方向对齐到气泡中心，偏移限制在一行高度内
        # textlines 按掩码质心排版，而 region_cy 来自外接矩形中心，两者往往不完全重合。用 y_offset 做有限的竖直微调。
        line_cy = np.array([line.pos_y for line in textlines]).mean() + line_height / 2
        region_cy = region_y + region_h / 2
        y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

        # 根据各行位置计算绘制画布的包围盒；同时在气泡坐标系里画出文字占位，并把行坐标转到画布局部坐标系
        lines_x1, lines_x2 = [], []
        for line in textlines:
            lines_x1.append(line.pos_x)  # 行左边界
            lines_x2.append(max(line.pos_x, 0) + line.length)  # 行右边界
        lines_x1 = np.array(lines_x1)
        lines_x2 = np.array(lines_x2)
        # 计算包住所有文字行（含描边）的最小画布矩形。
        canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
        canvas_y1, canvas_y2 = textlines[0].pos_y - sw, textlines[-1].pos_y + line_height + sw
        canvas_h = int(canvas_y2 - canvas_y1)
        canvas_w = int(canvas_x2 - canvas_x1)
        lines_map = np.zeros_like(ballon_mask, dtype=np.uint8)
        for line in textlines:
            # line.pos_y += y_offset
            # 为每一行文字画一个占位矩形，生成 lines_map，用来判断文字是否超出气泡。
            cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset), (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)
            # 从掩码坐标系转到画布局部坐标
            line.pos_x -= canvas_x1
            line.pos_y -= canvas_y1

        if verbose:
            # debug: 在 ballon_mask 上叠加外接矩形（红）与文字占位矩形（绿）
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'result', 'ballon_debug')
            stem = f'{xyxy[0]}_{xyxy[1]}_{xyxy[2]}_{xyxy[3]}'
            debug_mask_path = os.path.join(debug_dir, f'ballon_mask_{stem}.png')
            debug_base = cv2.imread(debug_mask_path, cv2.IMREAD_GRAYSCALE)
            if debug_base is not None:
                if rotated:
                    debug_base = np.array(Image.fromarray(debug_base).rotate(region.angle, expand=True))
                debug_vis = cv2.cvtColor(debug_base, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(debug_vis, (region_x, region_y), (region_x + region_w, region_y + region_h), (0, 0, 255), 2)
                if debug_vis.shape[:2] == lines_map.shape[:2]:
                    line_mask = lines_map > 0
                    debug_vis[line_mask] = (
                        debug_vis[line_mask].astype(np.float32) * 0.4
                        + np.array([0, 255, 0], dtype=np.float32) * 0.6
                    ).astype(np.uint8)
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(debug_mask_path, debug_vis)

        # 取该文本块的字体色 / 描边色，渲染出文字图层
        region_font_color, region_stroke_color = region.get_font_colors()
        textlines_image = render_lines(textlines, canvas_h, canvas_w, font_size, sw, line_spacing, region_font_color, region_stroke_color)

        # 计算文字画布中心相对于气泡提取区域的相对坐标（扣除旋转偏移与缩放）
        rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
        rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio

        # 比较文字占位与气泡有效区域：若文字超出气泡，提高缩放比（实际缩小绘制见下方注释）
        lines_area = np.sum(lines_map)
        lines_area += (max(0, region_y - canvas_y1) + max(0, canvas_y2 - region_h - region_y)) * canvas_w * 255 \
                        + (max(0, region_x - canvas_x1) + max(0, canvas_x2 - region_w - region_x)) * canvas_h * 255

        valid_lines_ratio = lines_area / np.sum(cv2.bitwise_and(lines_map, ballon_mask))
        if valid_lines_ratio > 1:  # 文字包围盒大于气泡有效区域
            resize_ratio = min(resize_ratio * valid_lines_ratio, (1 / downscale_constraint) ** 2)

        # 倾斜文本：把相对中心旋回原图方向，并把文字图层反旋转后裁掉空白
        if rotated:
            rcx = rel_cx * region_angle_cos - rel_cy * region_angle_sin
            rcy = rel_cx * region_angle_sin + rel_cy * region_angle_cos
            rel_cx = rcx
            rel_cy = rcy
            textlines_image = textlines_image.rotate(-region.angle, expand=True, resample=Image.BILINEAR)
            textlines_image = textlines_image.crop(textlines_image.getbbox())

        # 相对坐标还原为整图绝对坐标
        abs_cx = rel_cx + xyxy[0]
        abs_cy = rel_cy + xyxy[1]

        # 按 resize_ratio 缩小文字图层；实际使用中常把字号缩得过小，故已禁用
        if resize_ratio != 1:
            textlines_image = textlines_image.resize((int(textlines_image.width / resize_ratio), int(textlines_image.height / resize_ratio)))

        # 以文字图层中心对齐到绝对坐标，透明通道贴到原图
        abs_x = int(abs_cx - textlines_image.width / 2)
        abs_y = int(abs_cy - textlines_image.height / 2)
        img_pil.paste(textlines_image, (abs_x, abs_y), mask=textlines_image)
        # cv2.imshow('ballon_region', ballon_region)
        # cv2.imshow('cropped', original_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
        # cv2.imshow('raw_lines', np.array(raw_lines))
        # cv2.waitKey(0)

    return np.array(img_pil)
