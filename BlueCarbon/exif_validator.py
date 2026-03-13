# exif_validator.py
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime, timedelta

def get_exif(img_path_or_file):
    img = Image.open(img_path_or_file)
    exif_data = {}
    info = img._getexif()
    if not info:
        return {}
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_data[decoded] = value
    return exif_data

def has_gps(exif):
    if "GPSInfo" in exif:
        gps = exif["GPSInfo"]
        return True
    return False

def has_timestamp_recent(exif, max_age_minutes=60*24*7):  # default: 7 days
    # look for DateTimeOriginal or DateTime
    for key in ("DateTimeOriginal", "DateTime"):
        if key in exif:
            try:
                dt_str = exif[key]
                # format 'YYYY:MM:DD HH:MM:SS'
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                age = datetime.now() - dt
                return age <= timedelta(minutes=max_age_minutes)
            except Exception:
                return False
    return False

def basic_verify(img_path_or_file):
    exif = get_exif(img_path_or_file)
    reasons = []
    if not exif:
        reasons.append("No EXIF metadata")
    if not has_gps(exif):
        reasons.append("No GPS info")
    if not has_timestamp_recent(exif):
        reasons.append("Timestamp missing or too old")
    ok = len(reasons) == 0
    return ok, reasons

# Example usage:
if __name__ == "__main__":
    path = "example.jpg"
    ok, reasons = basic_verify(path)
    print("OK:", ok)
    print("Reasons:", reasons)
