import string
import easyocr


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}


def license_complies_format(text):

    if len(text) != 6:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.digits or text[1] in dict_char_to_int.keys()) and \
       (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) :
       
         return 1
    elif (text[0] in string.digits or text[0] in dict_char_to_int.keys()) and \
       (text[1] in string.digits or text[1] in dict_char_to_int.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
       (text[4] in string.digits or text[4] in dict_char_to_int.keys()) and \
       (text[5] in string.digits or text[5] in dict_char_to_int.keys()) :
       
         return 2
    else:
         return False



def format_license(text):

    # Initialize an empty string for the formatted license plate
    license_plate_ = ''
    plate_type = license_complies_format(text)
    



    # Format based on the detected type of license plate
    if plate_type == 1:  # Check the compliance format
        mapping = {0: dict_int_to_char, 1: dict_char_to_int, 4: dict_int_to_char, 5: dict_int_to_char,
                   2: dict_char_to_int, 3: dict_char_to_int}
    elif plate_type == 2:
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int,
                   2: dict_int_to_char, 3: dict_int_to_char}
    for j in range(len(text)):  # Iterate over each character
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]  # Convert using mapping
        else:
            license_plate_ += text[j]  # Keep the character as is


    return license_plate_


def read_license_plate(license_plate_crop):


    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace('-', '')

        if license_complies_format(text):
            return format_license(text), score
        
    return None, None


def get_car(license_plate, vehicle_track_ids):
    
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1