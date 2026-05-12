from PIL import Image


def detect_image_type(image):
    """
    Detect whether image is:
    - full body
    - upper body focused
    - lower body focused
    """

    width, height = image.size

    aspect_ratio = height / width

    # Tall image → likely full body
    if aspect_ratio > 1.8:
        return "full_body"

    # Medium portrait → likely upper/lower focus
    elif aspect_ratio > 1.3:
        return "partial_body"

    else:
        return "unknown"


def generate_region_crops(image):

    width, height = image.size

    image_type = detect_image_type(image)

    crops = {}

    # ------------------------------------------------
    # Full body image
    # ------------------------------------------------

    if image_type == "full_body":

        upper_crop = image.crop((
            0,
            0,
            width,
            int(height * 0.55)
        ))

        lower_crop = image.crop((
            0,
            int(height * 0.45),
            width,
            height
        ))

        crops["upper_body"] = upper_crop
        crops["lower_body"] = lower_crop
        crops["full_body"] = image

    # ------------------------------------------------
    # Partial body image
    # ------------------------------------------------

    elif image_type == "partial_body":

        # Use center heuristic

        upper_region = image.crop((
            0,
            0,
            width,
            int(height * 0.6)
        ))

        lower_region = image.crop((
            0,
            int(height * 0.4),
            width,
            height
        ))

        # Decide dominant region

        if height > width * 1.5:
            crops["lower_body"] = image
        else:
            crops["upper_body"] = image

        crops["full_body"] = image

    # ------------------------------------------------
    # Fallback
    # ------------------------------------------------

    else:

        crops["full_body"] = image

    return crops