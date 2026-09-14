import os
import sys
import cv2
import pytest
import numpy as np

from manga_translator.rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_reflow
from manga_translator.utils import (
    TextBlock,
    visualize_textblocks,
)


RENDER_IMAGE_FOLDER = 'test/testdata/render'
os.makedirs(RENDER_IMAGE_FOLDER, exist_ok=True)

def save_result(path, img, regions):
    path = os.path.join(RENDER_IMAGE_FOLDER, path)
    cv2.imwrite(path, visualize_textblocks(img, regions))


@pytest.mark.asyncio
async def test_default_renderer():
    width, height = 1000, 1000
    img = np.zeros((height, width, 3))
    regions = [
        TextBlock(
            [[[10, 10], [200, 10], [10, 400], [200, 400]]],
            texts=['a', 'b','c', 'd', 'e', 'f'],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[410, 10], [900, 10], [410, 800], [900, 800]]],
            texts=['eng', 'pne'],
            translation=#'aaaaaa bbbbbbbbbbbb cccc' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd eeeeeeeeeeeeee fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd ' \
                'normal english sentences can be hyphenated! ' \
                'Pneumonoultramicroscopicsilicovolcanoconiosis'
        ),
    ]
    for region in regions:
        region.target_lang = 'ENG'
        region.set_font_colors([255, 255, 255], [200, 200, 200])
        region.font_size = 100

    img_rendered = await dispatch_rendering(img, regions, hyphenate=False)
    save_result('default1.png', img_rendered, regions)


@pytest.mark.asyncio
async def test_reflow_renderer():
    width, height = 1000, 1000
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.ellipse(img, (150, 200), (140, 190), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(img, (650, 400), (240, 380), 0, 0, 360, (250, 250, 250), -1)
    regions = [
        TextBlock(
            [[[80, 80], [220, 80], [80, 320], [220, 320]]],
            texts=['a', 'b', 'c', 'd', 'e', 'f'],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[480, 80], [820, 80], [480, 720], [820, 720]]],
            texts=['eng', 'pne'],
            translation='normal english sentences can be hyphenated! '
                'Pneumonoultramicroscopicsilicovolcanoconiosis'
        ),
    ]
    for region in regions:
        region.target_lang = 'ENG'
        region.set_font_colors([20, 20, 20], [250, 250, 250])
        region.font_size = 40

    img_rendered = await dispatch_reflow(img.copy(), img, regions, hyphenate=True)
    save_result('reflow1.png', img_rendered, regions)
