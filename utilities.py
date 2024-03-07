import cv2

def draw_text(img, text,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0,0),
    font_scale=0.4,
    font_thickness=1,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0)
    ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y+ text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return text_size