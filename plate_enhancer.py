import cv2
from realesrgan import RealESRGANer

def enhance_with_realesrgan(image, scale=2):
    try:
        # Create ESRGANer without manual model injection
        upsampler = RealESRGANer(
            scale=scale,
            model_path=None,  # auto-download if not present
            model=None,       # use default RRDBNet internally
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        enhanced, _ = upsampler.enhance(image)
        return enhanced

    except Exception as e:
        print(f"[Real-ESRGAN Error] {e}")
        return image
