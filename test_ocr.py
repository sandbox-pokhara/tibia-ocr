import cv2

from tibia_ocr.hash_ocr import convert_line
from tibia_ocr.hash_ocr import convert_paragraph


def visualize(img, text):
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
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "exorirtdttt"

    image = cv2.imread("data/test1.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "BruiseBane"

    image = cv2.imread("data/test2.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "ttttttttttttttttttttffffffff"

    image = cv2.imread("data/test3.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))

    out = convert_paragraph(threshed)

    visualize(image, out)
    assert (
        out
        == "HitPoints830\n"
        "Mana305\n"
        "SoulPoints200\n"
        "Capacity587\n"
        "Speed\n"
        "Food0000\n"
        "Stamina3925"
    )

    image = cv2.imread("data/test4.png")
    threshed = cv2.inRange(image, (244, 244, 244), (244, 244, 244))
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "Tibia"

    image = cv2.imread("data/test5.png")
    threshed = cv2.inRange(image, (191, 191, 191), (191, 191, 191))
    out = convert_line(threshed)
    visualize(image, out)
    assert out == "28k"


def test_small_font():
    image = cv2.imread("data/test-small-1.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert out == "Reset\nOkApplyCancel"

    image = cv2.imread("data/test-small-2.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert out == "Bar1Bar2Bar3\nBar1Bar2Bar3\nBar1Bar2Bar3"

    image = cv2.imread("data/test-small-3.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert out == "CreateNewAccountLogin"

    image = cv2.imread("data/test-small-4.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert out == "OpenScreenshotFolder"

    image = cv2.imread("data/test-small-5.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert (
        out
        == "RuleViolations\n"
        "Manual\n"
        "FAQ\n"
        "In\n"
        "ImportTibia10Config\n"
        "ImportTibia10Map\n"
        "ImportFlashMap\n"
        "ExportAllOptions\n"
        "ExportMinimap\n"
        "ImportOptionsMinimap\n"
        "ResetAllOptions"
    )

    image = cv2.imread("data/test-small-6.png")
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    out = convert_paragraph(threshed, font="small")
    visualize(image, out)
    assert (
        out
        == "Controls\n"
        "GeneralHotkeys\n"
        "ActionBarHotkeys\n"
        "CustomHotkeys\n"
        "Interface\n"
        "Graphics\n"
        "Sound\n"
        "Misc"
    )
