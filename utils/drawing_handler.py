import cv2

def draw_polygons(frame, crosswalk_poly, risk_poly):
    cv2.polylines(frame, [crosswalk_poly], True, (0, 255, 0), 2)
    cv2.polylines(frame, [risk_poly], True, (0, 0, 255), 2)

