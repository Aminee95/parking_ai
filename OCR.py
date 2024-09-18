import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}

def license_complies_format(text):
    """Checks if the license plate format matches expected patterns."""
    
    if len(text) != 6:
        return False
    
    if all([text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys(),
            text[1] in string.digits or text[1] in dict_char_to_int.keys(),
            text[2] in string.digits or text[2] in dict_char_to_int.keys(),
            text[3] in string.digits or text[3] in dict_char_to_int.keys(),
            text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys(),
            text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()]):
        return 1
    elif all([text[0] in string.digits or text[0] in dict_char_to_int.keys(),
              text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys(),
              text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys(),
              text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys(),
              text[4] in string.digits or text[4] in dict_char_to_int.keys(),
              text[5] in string.digits or text[5] in dict_char_to_int.keys()]):
        return 2
    elif all([text[0] in string.digits or text[0] in dict_char_to_int.keys(),
              text[1] in string.digits or text[1] in dict_char_to_int.keys(),
              text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys(),
              text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys(),
              text[4] in string.digits or text[4] in dict_char_to_int.keys(),
              text[5] in string.digits or text[5] in dict_char_to_int.keys()]):
        return 3
    else:
        return False

def format_license(text):
    """Formats the license plate based on the detected pattern."""
    
    plate_type = license_complies_format(text)
    if not plate_type:
        return text
    
    mappings = {
        1: {0: dict_int_to_char, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_int_to_char, 5: dict_int_to_char},
        2: {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 3: dict_int_to_char, 4: dict_char_to_int, 5: dict_char_to_int},
        3: {0: dict_char_to_int, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_int_to_char, 4: dict_char_to_int, 5: dict_char_to_int}
    }
    
    license_plate_ = ''.join([mappings[plate_type][j].get(text[j], text[j]) for j in range(len(text))])
    
    return license_plate_

def read_license_plate(license_plate_crop):
    """Extracts and formats the license plate text from the image crop."""
    
    detections = reader.readtext(license_plate_crop)
    
    for bbox, text, score in detections:
        text = text.upper().replace('-', '')
        if license_complies_format(text):
            return format_license(text), score
    
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """Finds the car in the vehicle list that corresponds to the detected license plate."""
    
    x1, y1, x2, y2, score, class_id = license_plate
    
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    
    return -1, -1, -1, -1, -1
