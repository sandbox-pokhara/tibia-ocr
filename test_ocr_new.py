from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from tibia_ocr.hash_ocr import convert_line
from tibia_ocr.hash_ocr import convert_paragraph

color191 = np.array([191, 191, 191])
color192 = np.array([192, 192, 192])
color244 = np.array([244, 244, 244])


def visualize(img: MatLike, text: str):
    scaled = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_NEAREST,
    )
    scaled = cv2.copyMakeBorder(
        scaled,
        25 * (text.count("\n") + 1),
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    y0, dy = 20, 25
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * dy
        cv2.putText(scaled, line, (0, y), 3, 0.7, (0, 255, 0), 2)
    cv2.imshow("", scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_big_font():
    image = cv2.imread("data/test.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "exorid"

    image = cv2.imread("data/test1.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "BiseBane"

    image = cv2.imread("data/test2.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_line(threshed)
    visualize(image, out)
    assert out == ""

    image = cv2.imread("data/test3.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed)
    visualize(image, out)
    assert (
        out == "HPois830\n"
        "Mana305\n"
        "SoulPois200\n"
        "Capacity587\n"
        "Speed\n"
        "Food0000\n"
        "amina3925"
    )

    image = cv2.imread("data/test4.png")
    threshed = cv2.inRange(image, color244, color244)
    out = convert_line(threshed, debug=True)
    visualize(image, out)
    assert out == "ibia"

    image = cv2.imread("data/test5.png")
    threshed = cv2.inRange(image, color191, color191)
    out = convert_line(threshed, debug=True)
    visualize(image, out)
    assert out == "k"


def test_small_font():
    small = Path() / "tibia_ocr" / "fonts" / "small.json"

    image = cv2.imread("data/test-small-1.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert out == "\nOkApplyCancel"

    image = cv2.imread("data/test-small-2.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert out == "r1r2r3\n" "r1r2r3\n" "r1r2r3"

    image = cv2.imread("data/test-small-3.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert out == "CteNcountgin"

    image = cv2.imread("data/test-small-4.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert out == "OpenScnshotFolder"

    image = cv2.imread("data/test-small-5.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert (
        out == "RuleViolions\n"
        "Manual\n"
        "FAQ\n"
        "In\n"
        "ImpoTibia10Conig\n"
        "ImpoTibia10Map\n"
        "ImpoFlhMap\n"
        "ExpoAllOptions\n"
        "ExpoMinip\n"
        "ImpoOptionsMinip\n"
        "AllOptions"
    )

    image = cv2.imread("data/test-small-6.png")
    threshed = cv2.inRange(image, color192, color192)
    out = convert_paragraph(threshed, font=small)
    visualize(image, out)
    assert (
        out == "Contls\n"
        "neralHot\n"
        "tionrHotk\n"
        "CustomHot\n"
        "Interfe\n"
        "Graphics\n"
        "Sound\n"
        "Misc"
    )


test_small_font()
